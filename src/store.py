"""Shared embeddings + vector store helpers.

Kept in one place so ingestion and retrieval always use the SAME embedding
model and the SAME Chroma collection.
"""

from functools import lru_cache

import config


@lru_cache(maxsize=1)
def get_embeddings():
    """FastEmbed (ONNX) embeddings — small and torch-free, ideal for cloud."""
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name=config.EMBED_MODEL)


def get_vectorstore():
    """Return the persisted Chroma vector store handle."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.VECTOR_DIR,
        embedding_function=get_embeddings(),
    )
