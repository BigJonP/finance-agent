EMBEDDER_CONFIG = {
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "cache_folder": "./models/embedders/Qwen3-Embedding-0.6B",
    "model_kwargs": {
        "device": "cuda",
    },
    "encode_kwargs": {
        "batch_size": 64,
        "normalize_embeddings": True,
    },
}

VECTOR_STORE_CONFIG = {
    "collection_name": "documents",
    "persist_directory": "./chroma_db",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "batch_size": 64,
    "top_k": 5,
    "recursive_character_text_splitter": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "length_function": len,
        "separators": ["\n\n", "\n", " ", ""],
    },
    "embedder_config": EMBEDDER_CONFIG,
}
