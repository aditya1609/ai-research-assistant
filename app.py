"""Streamlit chat UI for the cloud Research Assistant.

Run locally:   streamlit run app.py
Deploy:        push to GitHub -> Streamlit Cloud / Hugging Face Spaces
               (set GROQ_API_KEY as a secret).
"""

import os

import streamlit as st

# Load .env when running locally (optional; ignored if python-dotenv missing).
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

# On Streamlit Cloud, secrets are exposed via st.secrets -> mirror into env vars
# so the rest of the code (which reads os.environ) works unchanged.
# Locally there's usually no secrets.toml, so accessing st.secrets can raise;
# we ignore that and rely on the .env file instead.
try:
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER", "GROQ_MODEL"):
        if key in st.secrets and key not in os.environ:
            os.environ[key] = str(st.secrets[key])
except Exception:  # noqa: BLE001  (no secrets.toml locally -> use .env)
    pass

import config  # noqa: E402  (import after env is populated)
from src import ingest  # noqa: E402
from src.agent import ask  # noqa: E402

st.set_page_config(page_title="AI Research Assistant", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Preparing the knowledge base…")
def _prepare_index():
    """Build the vector index once per server session if it isn't there yet."""
    return ingest.ensure_index()


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🔎 Research Assistant")
    st.caption("Agentic RAG · documents + web + calculator")

    provider = config.LLM_PROVIDER
    model = {
        "groq": config.GROQ_MODEL,
        "openai": config.OPENAI_MODEL,
        "ollama": config.OLLAMA_MODEL,
    }.get(provider, "?")
    st.write(f"**LLM:** `{provider}` — `{model}`")

    key_ok = (
        (provider == "groq" and os.environ.get("GROQ_API_KEY"))
        or (provider == "openai" and os.environ.get("OPENAI_API_KEY"))
        or (provider == "ollama")
    )
    if key_ok:
        st.success("API key detected ✅")
    else:
        st.error("Missing API key. Add GROQ_API_KEY to your secrets/.env.")

    try:
        n_chunks = _prepare_index()
        st.info(f"Indexed chunks: **{n_chunks}**")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Index error: {exc}")

    st.markdown("**Documents**")
    if os.path.isdir(config.DOCS_DIR):
        for name in sorted(os.listdir(config.DOCS_DIR)):
            if os.path.splitext(name)[1].lower() in config.SUPPORTED_EXTENSIONS:
                st.write(f"• {name}")

    if st.button("🔁 Rebuild index"):
        ingest.build_index(force=True)
        _prepare_index.clear()
        st.rerun()

    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Try: *How much PTO do employees get?* · "
               "*What does the Business plan cost per year?* · "
               "*What is RAG?*")


# ---------------- Chat ----------------
st.title("Personal Research Assistant")
st.caption("Ask about the sample documents, the web, or math. The agent picks the right tool.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Ask a question…")
if prompt:
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                history = st.session_state.messages[:-1]
                answer = ask(prompt, history=history)
            except Exception as exc:  # noqa: BLE001
                answer = f"⚠️ Error: {exc}"
        st.markdown(answer)

    st.session_state.messages.append(("assistant", answer))
