"""Document loaders for the sample docs (.md, .txt, .pdf).

Each file is turned into one or more LangChain `Document` objects with a
`source` metadata field so answers can be traced back to the file.
"""

import os
from typing import List

from langchain_core.documents import Document

import config


def _load_text_file(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": os.path.basename(path)})]


def _load_pdf(path: str) -> List[Document]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    docs: List[Document] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": os.path.basename(path), "page": i + 1},
                )
            )
    return docs


def load_file(path: str) -> List[Document]:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".md", ".txt"}:
        return _load_text_file(path)
    if ext == ".pdf":
        return _load_pdf(path)
    return []


def load_directory(docs_dir: str = None) -> List[Document]:
    """Load every supported file in `docs_dir` (recursively)."""
    docs_dir = docs_dir or config.DOCS_DIR
    documents: List[Document] = []
    if not os.path.isdir(docs_dir):
        return documents

    for root, _, files in os.walk(docs_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in config.SUPPORTED_EXTENSIONS:
                path = os.path.join(root, name)
                try:
                    documents.extend(load_file(path))
                except Exception as exc:  # noqa: BLE001
                    print(f"[loaders] Skipped {name}: {exc}")
    return documents
