import os
import json
import tempfile
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import mlflow
from mlflow.tracking import MlflowClient
from langchain_core.documents import Document

from .mlflow_config import get_mlflow_config, MLflowConfig

logger = logging.getLogger(__name__)


class RetrieverTracker:
    def __init__(self, config: Optional[MLflowConfig] = None):
        self.config = config or get_mlflow_config()
        try:
            self.client = MlflowClient(tracking_uri=self.config.tracking_uri)
            self._setup_experiment()
            self.current_run = None
        except Exception:
            self.client = None
            self.current_run = None

    def _setup_experiment(self):
        try:
            mlflow.set_tracking_uri(self.config.tracking_uri)

            experiment = self.client.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(
                    self.config.experiment_name,
                    artifact_location=self.config.artifact_location,
                )
            else:
                experiment_id = experiment.experiment_id

            mlflow.set_experiment(self.config.experiment_name)
            return experiment_id
        except Exception:
            return None

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ):
        if self.client is None:
            return None

        # Don't end existing runs if we're starting a nested run
        if self.current_run is not None and not nested:
            mlflow.end_run()

        run_name = (
            run_name
            or self.config.run_name
            or f"retriever-run-{datetime.now().isoformat()}"
        )
        tags = tags or {}
        tags.update({"component": "retriever", "timestamp": datetime.now().isoformat()})

        try:
            self.current_run = mlflow.start_run(
                run_name=run_name, tags=tags, nested=nested
            )
            return self.current_run
        except Exception as e:
            logger.error(f"Error starting run: {e}")
            pass

    def end_run(self):
        if self.client is None:
            return

        if self.current_run is not None:
            mlflow.end_run()
            self.current_run = None

    def log_vector_store_config(self, config: Dict[str, Any]):
        mlflow.log_params(
            {
                "collection_name": config.get("collection_name"),
                "chunk_size": config.get("chunk_size"),
                "chunk_overlap": config.get("chunk_overlap"),
                "batch_size": config.get("batch_size"),
                "top_k": config.get("top_k"),
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f, indent=2, default=str)
            mlflow.log_artifact(f.name, "vector_store_config.json")
            os.unlink(f.name)

    def log_embedder_config(self, config: Dict[str, Any]):
        mlflow.log_params(
            {
                "embedder_model": config.get("model_name"),
                "embedder_device": config.get("model_kwargs", {}).get("device"),
                "embedder_batch_size": config.get("encode_kwargs", {}).get(
                    "batch_size"
                ),
                "embedder_normalize": config.get("encode_kwargs", {}).get(
                    "normalize_embeddings"
                ),
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f, indent=2, default=str)
            mlflow.log_artifact(f.name, "embedder_config.json")
            os.unlink(f.name)

    def log_document_processing_metrics(
        self,
        total_documents: int,
        total_chunks: int,
        processing_time: float,
        chunk_sizes: List[int],
    ):
        mlflow.log_metrics(
            {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "processing_time_seconds": processing_time,
                "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes)
                if chunk_sizes
                else 0,
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            }
        )

        for i, size in enumerate(chunk_sizes[:10]):
            mlflow.log_metric(f"chunk_size_{i}", size)

    def log_search_metrics(
        self,
        query: str,
        top_k: int,
        search_time: float,
        results_count: int,
        scores: List[float],
    ):
        mlflow.log_metrics(
            {
                "search_time_seconds": search_time,
                "results_count": results_count,
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
            }
        )

        mlflow.log_params(
            {
                "search_top_k": top_k,
                "search_query_length": len(query),
            }
        )

        mlflow.log_text(query, "search_query.txt")

        if scores:
            for i, score in enumerate(scores[:10]):
                mlflow.log_metric(f"search_score_{i}", score)

    def log_vector_store_artifacts(self, persist_directory: str):
        if os.path.exists(persist_directory):
            mlflow.log_artifact(persist_directory, "vector_store")

    def log_sample_documents(self, documents: List[Document], max_samples: int = 10):
        sample_docs = documents[:max_samples]

        doc_summary = []
        for i, doc in enumerate(sample_docs):
            doc_summary.append(
                {
                    "index": i,
                    "page_content_length": len(doc.page_content),
                    "metadata": doc.metadata,
                }
            )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(doc_summary, f, indent=2, default=str)
            mlflow.log_artifact(f.name, "sample_documents_summary.json")
            os.unlink(f.name)

        for i, doc in enumerate(sample_docs):
            content = f"Document {i}:\n{'-' * 50}\n{doc.page_content}\n\nMetadata: {doc.metadata}"
            mlflow.log_text(content, f"sample_document_{i}.txt")

    def log_metrics(self, metrics: Dict[str, float]):
        if self.client is None:
            return
        try:
            mlflow.log_metrics(metrics)
        except Exception:
            pass

    def log_params(self, params: Dict[str, Any]):
        if self.client is None:
            return
        try:
            mlflow.log_params(params)
        except Exception:
            pass

    def log_error(self, error: Exception, context: str = ""):
        if self.client is None:
            return

        try:
            error_info = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "timestamp": datetime.now().isoformat(),
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(error_info, f, indent=2)
                mlflow.log_artifact(f.name, "error_log.json")
                os.unlink(f.name)

            mlflow.log_param("error_occurred", True)
            mlflow.log_param(
                f"error_type_{context}_{datetime.now().timestamp()}",
                error_info["error_type"],
            )
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
