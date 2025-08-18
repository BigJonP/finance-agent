import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


EMBEDDER_CONFIG = {
    "model_name": "BAAI/bge-base-en-v1.5",
    "cache_folder": os.path.join(PROJECT_ROOT, "./models/embedders/bge-base-en-v1.5"),
    "model_kwargs": {
        "device": "cuda",
    },
    "encode_kwargs": {
        "batch_size": 64,
        "normalize_embeddings": True,
    },
}


RECURSIVE_CHARACTER_TEXT_SPLITTER_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "length_function": len,
    "separators": ["\n\n", "\n", " ", ""],
}


VECTOR_STORE_CONFIG = {
    "collection_name": "documents",
    "persist_directory": os.path.join(PROJECT_ROOT, "chroma_db"),
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "batch_size": 64,
    "top_k": 5,
    "embedder_type": "local",
    "embedder_kwargs": EMBEDDER_CONFIG,
    "recursive_character_text_splitter": RECURSIVE_CHARACTER_TEXT_SPLITTER_CONFIG,
}
