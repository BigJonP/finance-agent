import asyncio
from functools import cache
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retriever.embedder import get_embedder


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
            chunk_size=config.get("recursive_character_text_splitter").get(
                "chunk_size"
            ),
            chunk_overlap=config.get("recursive_character_text_splitter").get(
                "chunk_overlap"
            ),
            length_function=config.get("recursive_character_text_splitter").get(
                "length_function"
            ),
            separators=config.get("recursive_character_text_splitter").get(
                "separators"
            ),
        )

    async def add_documents(self, documents: List[Document]) -> None:
        await self.add_documents_true_streaming(documents)

    async def add_documents_stream(self, documents: List[Document]) -> None:
        chunk_queue = asyncio.Queue(maxsize=100)

        consumer_task = asyncio.create_task(self._chunk_consumer(chunk_queue))
        producer_task = asyncio.create_task(
            self._chunk_producer(documents, chunk_queue)
        )

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

    async def _chunk_producer(
        self, documents: List[Document], chunk_queue: asyncio.Queue
    ) -> None:
        for document in documents:
            chunks = await self.text_splitter.atransform_documents([document])

            for chunk in chunks:
                await chunk_queue.put(chunk)

        await chunk_queue.put(None)

    async def search(
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


@cache
def get_vector_store(config: dict) -> AsyncVectorStore:
    return AsyncVectorStore(config)
