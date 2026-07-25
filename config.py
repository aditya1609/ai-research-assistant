"""Central configuration for the cloud Research Assistant.

Everything is controlled by environment variables so the same code runs
locally and on Streamlit Cloud / Hugging Face Spaces without edits.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Which LLM backend to use: "groq" (default, free) | "openai" | "ollama" ---
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

# Model per provider (override with env vars if you like)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Generation settings
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))

# --- Documents used for RAG ---
# By default we index the committed, non-confidential sample_docs folder.
DOCS_DIR = os.environ.get("DOCS_DIR", os.path.join(BASE_DIR, "sample_docs"))
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}

# --- Vector store (Chroma, persisted to disk) ---
VECTOR_DIR = os.environ.get("VECTOR_DIR", os.path.join(BASE_DIR, "chroma_db"))
COLLECTION_NAME = "docs"

# --- Embeddings (FastEmbed = ONNX, small, no torch -> deploys anywhere) ---
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# --- Chunking + retrieval ---
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))
TOP_K = int(os.environ.get("TOP_K", "5"))
