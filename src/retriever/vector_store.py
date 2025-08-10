import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retriever.embedder import get_embedder
from tracking import RetrieverTracker

_vector_store_instance = None
_vector_store_config = None


class AsyncVectorStore:
    def __init__(self, config: dict):
        self.config = config
        self.embeddings = get_embedder()
        self.vector_store = Chroma(
            collection_name=config.get("collection_name"),
            embedding_function=self.embeddings,
            persist_directory=config.get("persist_directory"),
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get("recursive_character_text_splitter").get("chunk_size"),
            chunk_overlap=config.get("recursive_character_text_splitter").get("chunk_overlap"),
            length_function=config.get("recursive_character_text_splitter").get("length_function"),
            separators=config.get("recursive_character_text_splitter").get("separators"),
        )

        self.tracker = RetrieverTracker()
        self._log_configs()

    def _log_configs(self):
        try:
            self.tracker.start_run(run_name="vector-store-init")
            self.tracker.log_vector_store_config(self.config)
            self.tracker.log_embedder_config(self.config.get("embedder_config", {}))
            self.tracker.end_run()
        except Exception as e:
            print(f"Warning: Failed to log configs to MLflow: {e}")

    async def add_documents(self, documents: List[Document]) -> None:
        start_time = time.time()
        try:
            self.tracker.start_run(run_name="document-processing")
            self.tracker.log_sample_documents(documents)

            await self.add_documents_stream(documents)

            processing_time = time.time() - start_time
            estimated_chunks = len(documents) * 2
            chunk_sizes = [len(doc.page_content) for doc in documents]

            self.tracker.log_document_processing_metrics(
                total_documents=len(documents),
                total_chunks=estimated_chunks,
                processing_time=processing_time,
                chunk_sizes=chunk_sizes,
            )
        except Exception as e:
            self.tracker.log_error(e, "document_processing")
            raise
        finally:
            self.tracker.end_run()

    async def add_documents_stream(self, documents: List[Document]) -> None:
        chunk_queue = asyncio.Queue(maxsize=100)
        consumer_task = asyncio.create_task(self._chunk_consumer(chunk_queue))
        producer_task = asyncio.create_task(self._chunk_producer(documents, chunk_queue))

        await asyncio.gather(producer_task, consumer_task)

    async def _chunk_consumer(self, chunk_queue: asyncio.Queue) -> None:
        batch = []
        batch_size = self.config.get("batch_size")

        while True:
            chunk = await chunk_queue.get()
            if chunk is None:
                break

            batch.append(chunk)

            if len(batch) >= batch_size:
                await self._add_batch_async(batch)
                batch = []

        if batch:
            await self._add_batch_async(batch)

    async def _add_batch_async(self, documents: List[Document]) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.vector_store.add_documents, documents)

    async def _chunk_producer(self, documents: List[Document], chunk_queue: asyncio.Queue) -> None:
        for document in documents:
            chunks = await self.text_splitter.atransform_documents([document])

            for chunk in chunks:
                await chunk_queue.put(chunk)

        await chunk_queue.put(None)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        start_time = time.time()

        try:
            self.tracker.start_run(run_name="search-query")

            if top_k is None:
                top_k = self.config.get("top_k")

            results = self.vector_store.similarity_search_with_score(
                query,
                top_k,
                filter_dict,
            )
            search_time = time.time() - start_time

            scores = [score for _, score in results]
            self.tracker.log_search_metrics(
                query=query,
                top_k=top_k,
                search_time=search_time,
                results_count=len(results),
                scores=scores,
            )

            return results

        except Exception as e:
            self.tracker.log_error(e, "search_query")
            raise
        finally:
            self.tracker.end_run()


def get_vector_store(config: dict) -> AsyncVectorStore:
    global _vector_store_instance, _vector_store_config

    if _vector_store_instance is not None and _vector_store_config == config:
        return _vector_store_instance

    _vector_store_instance = AsyncVectorStore(config)
    _vector_store_config = config
    return _vector_store_instance
