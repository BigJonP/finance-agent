from functools import cache

from langchain_huggingface import HuggingFaceEmbeddings

from retriever.config import EMBEDDER_CONFIG


@cache
def get_local_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(**EMBEDDER_CONFIG)
