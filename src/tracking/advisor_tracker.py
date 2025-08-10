import os
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict

import mlflow
from mlflow.tracking import MlflowClient

from .mlflow_config import get_mlflow_config, MLflowConfig
from api.services.config import (
    SYSTEM_PROMPT,
    MODEL_NAME,
    RETRIEVER_K,
    RELEVANT_DOCUMENTS_TOP_K,
    MAX_TOKENS,
    TEMPERATURE,
)


class AdvisorTracker:
    def __init__(self, config: Optional[MLflowConfig] = None):
        self.config = config or get_mlflow_config()
        self.client = MlflowClient(tracking_uri=self.config.tracking_uri)
        self._setup_experiment()
        self.current_run = None

    def _setup_experiment(self):
        mlflow.set_tracking_uri(self.config.tracking_uri)

        experiment = self.client.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            experiment_id = self.client.create_experiment(
                self.config.experiment_name, artifact_location=self.config.artifact_location
            )
        else:
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(self.config.experiment_name)
        return experiment_id

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        if self.current_run is not None:
            mlflow.end_run()

        run_name = run_name or self.config.run_name or f"advisor-run-{datetime.now().isoformat()}"
        tags = tags or {}
        tags.update({"component": "financial_advisor", "timestamp": datetime.now().isoformat()})

        self.current_run = mlflow.start_run(run_name=run_name, tags=tags)
        return self.current_run

    def end_run(self):
        if self.current_run is not None:
            mlflow.end_run()
            self.current_run = None

    def log_advisor_config(self):
        config_params = {
            "model_name": MODEL_NAME,
            "retriever_k": RETRIEVER_K,
            "relevant_documents_top_k": RELEVANT_DOCUMENTS_TOP_K,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }

        mlflow.log_params(config_params)

        mlflow.log_text(SYSTEM_PROMPT, "system_prompt.txt")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_params, f, indent=2)
            mlflow.log_artifact(f.name, "advisor_config.json")
            os.unlink(f.name)

    def log_portfolio_context(
        self,
        user_id: int,
        holdings: List[Dict[str, Any]],
        relevant_documents: List[str],
        processing_time: float,
    ):
        mlflow.log_params(
            {
                "user_id": user_id,
                "holdings_count": len(holdings),
                "relevant_documents_count": len(relevant_documents),
            }
        )

        mlflow.log_metrics(
            {
                "portfolio_context_processing_time_seconds": processing_time,
                "avg_document_length": (
                    sum(len(doc) for doc in relevant_documents) / len(relevant_documents)
                    if relevant_documents
                    else 0
                ),
            }
        )

        holdings_summary = []
        for holding in holdings:
            holdings_summary.append(
                {"stock": holding.get("stock", "Unknown"), "metadata_keys": list(holding.keys())}
            )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(holdings_summary, f, indent=2)
            mlflow.log_artifact(f.name, "holdings_summary.json")
            os.unlink(f.name)

        if relevant_documents:
            sample_docs = relevant_documents[:3]
            for i, doc in enumerate(sample_docs):
                mlflow.log_text(doc, f"relevant_document_{i}.txt")

    def log_advice_generation(
        self,
        user_prompt: str,
        advice_response: str,
        generation_time: float,
        token_count: Optional[int] = None,
    ):
        mlflow.log_metrics(
            {
                "advice_generation_time_seconds": generation_time,
                "user_prompt_length": len(user_prompt),
                "advice_response_length": len(advice_response),
            }
        )

        if token_count:
            mlflow.log_metric("response_token_count", token_count)

        mlflow.log_text(user_prompt, "user_prompt.txt")
        mlflow.log_text(advice_response, "advice_response.txt")

        prompt_analysis = {
            "prompt_word_count": len(user_prompt.split()),
            "prompt_contains_portfolio": "portfolio" in user_prompt.lower(),
            "prompt_contains_market": "market" in user_prompt.lower(),
            "response_word_count": len(advice_response.split()),
            "response_contains_recommendations": any(
                word in advice_response.lower()
                for word in ["recommend", "suggest", "consider", "advise"]
            ),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(prompt_analysis, f, indent=2)
            mlflow.log_artifact(f.name, "prompt_analysis.json")
            os.unlink(f.name)

    def log_vector_store_search(
        self, stock: str, enhanced_query: str, search_results: List[tuple], search_time: float
    ):
        mlflow.log_params(
            {
                "search_stock": stock,
                "search_query_length": len(enhanced_query),
                "search_results_count": len(search_results),
            }
        )

        mlflow.log_metrics(
            {
                "vector_search_time_seconds": search_time,
            }
        )

        mlflow.log_text(enhanced_query, f"search_query_{stock}.txt")

        if search_results:
            scores = [score for _, score in search_results]
            mlflow.log_metrics(
                {
                    f"search_score_avg_{stock}": sum(scores) / len(scores),
                    f"search_score_max_{stock}": max(scores),
                    f"search_score_min_{stock}": min(scores),
                }
            )

    def log_error(self, error: Exception, context: str = "", user_id: Optional[int] = None):
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(error_info, f, indent=2)
            mlflow.log_artifact(f.name, "advisor_error_log.json")
            os.unlink(f.name)

        mlflow.log_param("error_occurred", True)
        mlflow.log_param("error_type", error_info["error_type"])
        if user_id:
            mlflow.log_param("error_user_id", user_id)

    def log_metrics(self, metrics: Dict[str, float]):
        mlflow.log_metrics(metrics)

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
