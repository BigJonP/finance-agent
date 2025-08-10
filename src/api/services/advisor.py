import os
import logging
import time
from typing import List, Dict, Any
from openai import AsyncOpenAI

from api.models.schema import AdviceRequest, AdviceResponse
from db.db_util import get_user_holdings, get_stock_metadata
from retriever.vector_store import get_vector_store
from retriever.config import VECTOR_STORE_CONFIG
from tracking.advisor_tracker import AdvisorTracker
from api.services.config import (
    SYSTEM_PROMPT,
    ERROR_ADVICE,
    MODEL_NAME,
    RETRIEVER_K,
    RELEVANT_DOCUMENTS_TOP_K,
    MAX_TOKENS,
    TEMPERATURE,
)


class FinancialAdvisor:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store = get_vector_store(VECTOR_STORE_CONFIG)
        self.tracker = AdvisorTracker()

    def _build_enhanced_search_query(self, stock: str, metadata: Dict[str, Any] = None) -> str:
        query_parts = [stock]

        if metadata:
            if metadata.get("company_name"):
                query_parts.append(metadata["company_name"])

            if metadata.get("short_name"):
                query_parts.append(metadata["short_name"])

            if metadata.get("industries"):
                query_parts.append(f"industry {metadata['industries']}")

            if metadata.get("description"):
                desc_words = metadata["description"].split()[:25]
                query_parts.append(" ".join(desc_words))

        query_parts.extend(["financial analysis", "market trends", "investment"])

        return " ".join(query_parts)

    async def get_user_portfolio_context(
        self, user_id: int
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        start_time = time.time()

        try:
            with self.tracker.start_run(run_name=f"portfolio-context-{user_id}") as run:
                self.tracker.log_advisor_config()

                holdings = await get_user_holdings(user_id)
                if not holdings:
                    self.tracker.log_params({"holdings_found": False})
                    return [], []

                stock_symbols = [holding.get("stock", "") for holding in holdings]

                relevant_documents = []
                for stock in stock_symbols:
                    try:
                        stock_metadata = await get_stock_metadata(stock)
                        enhanced_query = self._build_enhanced_search_query(stock, stock_metadata)
                    except Exception as metadata_error:
                        logging.info(f"Error getting stock metadata: {metadata_error}")
                        enhanced_query = self._build_enhanced_search_query(stock, None)

                    search_start_time = time.time()
                    search_results = self.vector_store.search(
                        query=enhanced_query, top_k=RETRIEVER_K
                    )
                    search_time = time.time() - search_start_time

                    self.tracker.log_vector_store_search(
                        stock=stock,
                        enhanced_query=enhanced_query,
                        search_results=search_results,
                        search_time=search_time,
                    )

                    for doc, score in search_results:
                        if score > 0.9:
                            relevant_documents.append(doc.page_content)

                processing_time = time.time() - start_time

                self.tracker.log_portfolio_context(
                    user_id=user_id,
                    holdings=holdings,
                    relevant_documents=relevant_documents,
                    processing_time=processing_time,
                )

                return holdings, relevant_documents

        except Exception as e:
            processing_time = time.time() - start_time
            logging.info(f"Error getting portfolio context: {str(e)}")
            self.tracker.log_error(e, "portfolio_context_generation", user_id)
            return [], []

    async def generate_financial_advice(
        self, holdings: List[Dict[str, Any]], relevant_documents: List[str]
    ) -> str:
        start_time = time.time()

        try:
            portfolio_summary = self._format_portfolio_summary(holdings)

            market_context = "\n".join(relevant_documents[:RELEVANT_DOCUMENTS_TOP_K])

            user_prompt = f"""
Current Portfolio:
{portfolio_summary}

Relevant Market Information:
{market_context}

Please provide comprehensive financial advice based on the above information.
"""

            response = await self.openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )

            advice_response = response.choices[0].message.content
            generation_time = time.time() - start_time

            self.tracker.log_advice_generation(
                user_prompt=user_prompt,
                advice_response=advice_response,
                generation_time=generation_time,
                token_count=(
                    response.usage.total_tokens if hasattr(response.usage, "total_tokens") else None
                ),
            )

            return advice_response

        except Exception as e:
            generation_time = time.time() - start_time
            error_response = "I apologize, but I'm currently unable to generate personalized advice\n\nHowever, I can suggest some general principles: diversify your portfolio, consider your risk tolerance, and regularly review your investment strategy."

            self.tracker.log_error(e, "advice_generation")
            self.tracker.log_advice_generation(
                user_prompt=(
                    user_prompt
                    if "user_prompt" in locals()
                    else "Error occurred before prompt generation"
                ),
                advice_response=error_response,
                generation_time=generation_time,
            )

            return error_response

    def _format_portfolio_summary(self, holdings: List[Dict[str, Any]]) -> str:
        if not holdings:
            return "No current holdings found."

        summary = "Current Holdings:\n"
        for i, holding in enumerate(holdings, 1):
            stock = holding.get("stock", "Unknown")
            summary += f"{i}. {stock}\n"

        summary += f"\nTotal Holdings: {len(holdings)} stocks"
        return summary


_advisor_instance = None


def get_advisor() -> FinancialAdvisor:
    global _advisor_instance
    if _advisor_instance is None:
        _advisor_instance = FinancialAdvisor()
    return _advisor_instance


async def generate_advice(request: AdviceRequest) -> AdviceResponse:
    advisor = get_advisor()

    try:
        holdings, relevant_documents = await advisor.get_user_portfolio_context(request.user_id)

        advice = await advisor.generate_financial_advice(holdings, relevant_documents)

        return AdviceResponse(advice=advice)

    except Exception as e:
        logging.info(f"Error generating advice: {str(e)}")
        advisor.tracker.log_error(e, "main_advice_generation", request.user_id)
        return AdviceResponse(advice=ERROR_ADVICE)
