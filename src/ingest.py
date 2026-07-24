"""Build (or rebuild) the Chroma vector index from the documents folder.

Run directly:      python -m src.ingest
Rebuild from app:  ingest.build_index(force=True)
Auto-build:        ingest.ensure_index()   # builds only if empty/missing
"""

import os
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from src.loaders import load_directory
from src.store import get_vectorstore


def _split(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_index(force: bool = False) -> int:
    """Load docs, chunk, embed, and persist. Returns number of chunks indexed."""
    if force and os.path.isdir(config.VECTOR_DIR):
        shutil.rmtree(config.VECTOR_DIR, ignore_errors=True)

    documents = load_directory()
    if not documents:
        print(f"[ingest] No documents found in {config.DOCS_DIR}")
        return 0

    chunks = _split(documents)
    vs = get_vectorstore()
    vs.add_documents(chunks)
    print(f"[ingest] Indexed {len(chunks)} chunks from {len(documents)} document parts.")
    return len(chunks)


def _is_empty() -> bool:
    try:
        return get_vectorstore()._collection.count() == 0
    except Exception:  # noqa: BLE001
        return True


def ensure_index() -> int:
    """Build the index only if it doesn't exist yet. Returns chunk count."""
    if _is_empty():
        return build_index(force=True)
    return get_vectorstore()._collection.count()


if __name__ == "__main__":
    build_index(force=True)
