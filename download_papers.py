"""One-time helper: download a themed set of public AI research papers (PDFs)
from arXiv into the sample_docs/ folder, so the RAG assistant has real content.

Run once:
    python download_papers.py

Then rebuild the index (or just start the app, which auto-builds):
    python -m src.ingest
    streamlit run app.py

All papers are open-access on arXiv. Edit the PAPERS list to pick your own.
"""

import os
import ssl
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "sample_docs")

# (arXiv id, friendly filename) — feel free to add/remove any.
PAPERS = [
    ("1706.03762", "attention_is_all_you_need.pdf"),          # Transformers
    ("2005.11401", "retrieval_augmented_generation.pdf"),     # RAG (this project!)
    ("2106.09685", "lora_low_rank_adaptation.pdf"),           # Efficient fine-tuning
    # ("1810.04805", "bert.pdf"),                             # BERT (uncomment to add)
    # ("2005.14165", "gpt3_few_shot_learners.pdf"),          # GPT-3 (large file)
]

USER_AGENT = "Mozilla/5.0 (research-assistant-downloader)"


def download(arxiv_id: str, filename: str) -> None:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    dest = os.path.join(DOCS_DIR, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {filename} already exists")
        return

    print(f"[download] {url} -> {filename}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    # Some corporate networks need relaxed SSL; try normal first, then fallback.
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
    except ssl.SSLError:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r, open(dest, "wb") as f:
            f.write(r.read())
    kb = os.path.getsize(dest) // 1024
    print(f"[ok] saved {filename} ({kb} KB)")


def main() -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)
    for arxiv_id, filename in PAPERS:
        try:
            download(arxiv_id, filename)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] could not download {filename}: {exc}")
    print("\nDone. Now run:  python -m src.ingest   (or just: streamlit run app.py)")


if __name__ == "__main__":
    main()
