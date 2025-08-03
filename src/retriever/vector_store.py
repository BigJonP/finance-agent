import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.retriever.embedder import get_embedder


class AsyncVectorStore:
    def __init__(self, config: dict):
        self.config = config
        self.embeddings = get_embedder()
        self.vector_store = Chroma(
            collection_name=config.get("collection_name"),
            embedding_function=self.embeddings,
            persist_directory=config.get("persist_directory"),
        )
        self.executor = ThreadPoolExecutor(max_workers=config.get("max_workers"))

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get("recursive_character_text_splitter").get("chunk_size"),
            chunk_overlap=config.get("recursive_character_text_splitter").get("chunk_overlap"),
            length_function=config.get("recursive_character_text_splitter").get("length_function"),
            separators=config.get("recursive_character_text_splitter").get("separators"),
        )

    async def add_documents(self, documents: List[Document]) -> None:
        all_chunks = await self.text_splitter.atransform_documents(
            documents, batch_size=self.config.get("batch_size")
        )

        await self._add_chunks_parallel(all_chunks)

    async def _add_chunks_parallel(self, chunks: List[Document]) -> None:
        batches = [
            chunks[i : i + self.config.get("batch_size")]
            for i in range(0, len(chunks), self.config.get("batch_size"))
        ]

        tasks = [self._add_batch_async(batch) for batch in batches]

        await asyncio.gather(*tasks)

    async def _add_batch_async(self, documents: List[Document]) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.vector_store.add_documents, documents)

    async def search(
        self, query: str, top_k: Optional[int] = None, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        loop = asyncio.get_event_loop()

        if top_k is None:
            top_k = self.config.get("top_k")

        results = await loop.run_in_executor(
            self.executor,
            self.vector_store.similarity_search_with_score,
            query,
            top_k,
            filter_dict,
        )

        return results

    async def close(self):
        self.executor.shutdown(wait=True)


def get_vector_store(config: dict) -> AsyncVectorStore:
    return AsyncVectorStore(config)
