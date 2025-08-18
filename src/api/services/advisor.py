import os
import logging
import time
from typing import List, Dict, Any
from openai import OpenAI

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
)

logger = logging.getLogger(__name__)


class FinancialAdvisor:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store = get_vector_store(VECTOR_STORE_CONFIG)

        try:
            self.tracker = AdvisorTracker()
        except Exception as e:
            logger.error(f"Error initializing advisor tracker: {e}")
            pass

    def _build_enhanced_search_query(self, stock: str, metadata: Dict[str, Any] = None) -> str:
        query_parts = [stock]

        if metadata:
            if metadata.get("company_name"):
                query_parts.append(metadata["company_name"])

            if metadata.get("short_name"):
                query_parts.append(metadata["short_name"])

            if metadata.get("industries"):
                industries = metadata["industries"].split(",")
                for industry in industries:
                    industry = industry.strip()
                    if industry:
                        query_parts.append(industry)

            if metadata.get("description"):
                desc_words = metadata["description"].split()[:30]
                query_parts.append(" ".join(desc_words))

        query_parts.extend(
            [
                "financial analysis",
                "market trends",
                "investment",
                "stock analysis",
                "earnings",
                "revenue",
                "growth",
                "performance",
            ]
        )

        final_query = " ".join(query_parts)
        return final_query

    async def get_user_portfolio_context(
        self, user_id: int
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        start_time = time.time()

        try:
            if self.tracker and self.tracker.client:
                try:
                    import mlflow

                    with mlflow.start_run(run_name=f"portfolio-context-{user_id}"):
                        self.tracker.log_advisor_config()
                except Exception as mlflow_error:
                    logger.warning(f"MLflow tracking failed: {mlflow_error}")
            else:
                pass

            holdings = await get_user_holdings(user_id)

            if not holdings:
                if self.tracker and self.tracker.client:
                    self.tracker.log_params({"holdings_found": False})
                logger.info(
                    f"No holdings found for user {user_id}, will use general market context"
                )
                relevant_documents = await self._get_general_market_context()
                return [], relevant_documents

            stock_symbols = [holding.get("stock", "") for holding in holdings]

            relevant_documents = []
            for stock in stock_symbols:
                try:
                    stock_metadata = await get_stock_metadata(stock)
                    enhanced_query = self._build_enhanced_search_query(stock, stock_metadata)
                except Exception:
                    enhanced_query = self._build_enhanced_search_query(stock, None)

                search_start_time = time.time()

                search_results = self.vector_store.search(query=enhanced_query, top_k=RETRIEVER_K)
                search_time = time.time() - search_start_time

                logger.info(
                    f"Search for {stock} took {search_time:.3f}s, found {len(search_results)} results"
                )

                for i, (doc, score) in enumerate(search_results):
                    logger.debug(
                        f"  {stock} result {i+1}: score={score:.3f}, content_preview={doc.page_content[:100]}..."
                    )

                for doc, score in search_results:
                    if score > 0.7:
                        relevant_documents.append(doc.page_content)
                        logger.debug(f"Added document for {stock} with score {score:.3f}")
                    else:
                        logger.debug(
                            f"Skipped document for {stock} with score {score:.3f} (below threshold)"
                        )

            if not relevant_documents:
                logger.info("No stock-specific documents found, using general market context")
                relevant_documents = await self._get_general_market_context()

            processing_time = time.time() - start_time

            if self.tracker and self.tracker.client:
                self.tracker.log_portfolio_context(
                    user_id=user_id,
                    holdings=holdings,
                    relevant_documents=relevant_documents,
                    processing_time=processing_time,
                )

            return holdings, relevant_documents

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in get_user_portfolio_context: {e}")

            try:
                if self.tracker and self.tracker.client:
                    self.tracker.log_error(e, "portfolio_context_generation", user_id)
            except Exception as tracker_error:
                logger.error(
                    f"Error logging error in portfolio context generation: {tracker_error}"
                )

            relevant_documents = await self._get_general_market_context()
            return [], relevant_documents

    async def _get_general_market_context(self) -> List[str]:
        try:
            general_queries = [
                "financial analysis market trends investment",
                "portfolio diversification investment strategy",
                "market analysis economic outlook",
                "investment advice financial planning",
            ]

            relevant_documents = []
            for query in general_queries:
                try:
                    search_results = self.vector_store.search(query=query, top_k=2)

                    for doc, score in search_results:
                        if score > 0.7:
                            relevant_documents.append(doc.page_content)

                except Exception as e:
                    logger.warning(f"Error searching for general context with query '{query}': {e}")
                    continue

            unique_docs = list(dict.fromkeys(relevant_documents))
            return unique_docs[:RELEVANT_DOCUMENTS_TOP_K]

        except Exception as e:
            logger.error(f"Error getting general market context: {e}")
            return []

    async def generate_financial_advice(
        self, holdings: List[Dict[str, Any]], relevant_documents: List[str]
    ) -> str:
        start_time = time.time()

        try:
            portfolio_summary = self._format_portfolio_summary(holdings)

            market_context = "\n".join(relevant_documents[:RELEVANT_DOCUMENTS_TOP_K])

            user_prompt = f"""{SYSTEM_PROMPT}

Current Portfolio:
{portfolio_summary}

Relevant Reddit Posts:
{market_context}

Please provide comprehensive financial advice based on the above information.
"""

            response = self.openai_client.responses.create(
                model=MODEL_NAME,
                input=user_prompt,
                max_output_tokens=MAX_TOKENS,
            )

            advice_response = response.output_text
            generation_time = time.time() - start_time

            if self.tracker and self.tracker.client:
                self.tracker.log_advice_generation(
                    user_prompt=user_prompt,
                    advice_response=advice_response,
                    generation_time=generation_time,
                    token_count=(
                        response.usage.total_tokens
                        if hasattr(response.usage, "total_tokens")
                        else None
                    ),
                )

            return advice_response

        except Exception as e:
            generation_time = time.time() - start_time

            error_response = "I apologize, but I'm currently unable to generate personalized advice\n\nHowever, I can suggest some general principles: diversify your portfolio, consider your risk tolerance, and regularly review your investment strategy."

            try:
                if self.tracker and self.tracker.client:
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
            except Exception as tracker_error:
                logger.error(f"Error logging error in advice generation: {tracker_error}")

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
    try:
        logger.info(f"Starting advice generation for user {request.user_id}")

        advisor = get_advisor()
        holdings, relevant_documents = await advisor.get_user_portfolio_context(request.user_id)

        logger.info(
            f"Retrieved {len(holdings)} holdings and {len(relevant_documents)} relevant documents for user {request.user_id}"
        )

        if not relevant_documents:
            logger.warning(
                f"No relevant documents found for user {request.user_id}, advice may be generic"
            )

        advice = await advisor.generate_financial_advice(holdings, relevant_documents)

        logger.info(f"Successfully generated advice for user {request.user_id}")
        return AdviceResponse(advice=advice)

    except Exception as e:
        logger.error(f"Error generating advice for user {request.user_id}: {e}")

        try:
            advisor = get_advisor()
            if advisor.tracker and advisor.tracker.client:
                advisor.tracker.log_error(e, "main_advice_generation", request.user_id)
        except Exception as tracker_error:
            logger.error(f"Error logging error in main advice generation: {tracker_error}")

        return AdviceResponse(advice=ERROR_ADVICE)
