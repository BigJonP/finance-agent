import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retriever.embedder import get_embedder

logger = logging.getLogger(__name__)

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

        self._log_configs()

    def _log_configs(self):
        try:
            logger.info(f"Vector store initialized with config: {self.config}")
            logger.info(f"Embedder config: {self.config.get('embedder_config', {})}")
        except Exception as e:
            logger.warning(f"Failed to log configs to MLflow: {e}")

    async def add_documents(self, documents: List[Document]) -> None:
        initial_count = self.get_document_count()

        try:
            logger.info(
                f"Starting to add {len(documents)} documents. Initial count: {initial_count}"
            )
            await self.add_documents_stream(documents)

            final_count = self.get_document_count()

            logger.info(
                f"Document processing completed. Final count: {final_count}, Added: {final_count - initial_count}"
            )

        except Exception as e:
            logger.error(f"Error during document processing: {e}")
            raise
        finally:
            pass

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
        logger.info(f"Adding batch of {len(documents)} documents to vector store")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.vector_store.add_documents, documents)
        logger.info(f"Successfully added batch of {len(documents)} documents")

    async def _chunk_producer(self, documents: List[Document], chunk_queue: asyncio.Queue) -> None:
        total_chunks = 0
        for document in documents:
            chunks = await self.text_splitter.atransform_documents([document])
            total_chunks += len(chunks)

            for chunk in chunks:
                await chunk_queue.put(chunk)

        await chunk_queue.put(None)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        if top_k is None:
            top_k = self.config.get("top_k")

        results = self.vector_store.similarity_search_with_score(
            query,
            top_k,
            filter_dict,
        )
        return results

    def search_with_embedding(
        self,
        embedding: list[float],
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        results = self.vector_store.similarity_search_by_vector(embedding, top_k, filter_dict)
        return results

    def get_document_count(self) -> int:
        try:
            collection = self.vector_store._collection
            if collection:
                return collection.count()
            return 0
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0


def get_vector_store(config: dict) -> AsyncVectorStore:
    global _vector_store_instance, _vector_store_config

    if _vector_store_instance is not None and _vector_store_config == config:
        return _vector_store_instance

    _vector_store_instance = AsyncVectorStore(config)
    _vector_store_config = config
    return _vector_store_instance
