"""Streamlit chat UI for the cloud Research Assistant.

Run locally:   streamlit run app.py
Deploy:        push to GitHub -> Streamlit Cloud / Hugging Face Spaces
               (set GROQ_API_KEY as a secret).

Created by Aditya Raj.
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

CREATOR = "Aditya Raj"

st.set_page_config(
    page_title="AI Research Assistant · by Aditya Raj",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Styling ----------------
st.markdown(
    """
    <style>
      /* App background */
      .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, #eef2ff 0%, rgba(238,242,255,0) 55%),
                    radial-gradient(1000px 500px at 100% 0%, #f0fdfa 0%, rgba(240,253,250,0) 50%),
                    #ffffff;
      }
      /* Hero header */
      .hero {
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 45%, #0ea5e9 100%);
        border-radius: 20px;
        padding: 28px 32px;
        color: #ffffff;
        box-shadow: 0 12px 30px rgba(79,70,229,0.28);
        margin-bottom: 8px;
      }
      .hero h1 { color:#fff; margin:0; font-size:2.1rem; font-weight:800; letter-spacing:-0.5px; }
      .hero p  { color:#e9e7ff; margin:.4rem 0 0; font-size:1.02rem; }
      .hero .byline {
        display:inline-block; margin-top:14px; padding:6px 14px;
        background: rgba(255,255,255,0.16); border:1px solid rgba(255,255,255,0.35);
        border-radius: 999px; font-size:.86rem; font-weight:600;
      }
      .pills { margin-top:14px; }
      .pill {
        display:inline-block; margin:4px 6px 0 0; padding:5px 12px;
        background: rgba(255,255,255,0.14); border:1px solid rgba(255,255,255,0.28);
        border-radius: 999px; font-size:.8rem; color:#fff;
      }
      /* Sidebar card */
      .side-card {
        background:#ffffff; border:1px solid #e5e7eb; border-radius:14px;
        padding:14px 16px; margin-bottom:12px; box-shadow:0 2px 10px rgba(2,6,23,0.05);
      }
      .side-title { font-weight:800; font-size:1.05rem; margin:0; color:#111827; }
      .side-sub { color:#6b7280; font-size:.82rem; margin:.2rem 0 0; }
      .status-ok  { color:#047857; font-weight:600; }
      .status-bad { color:#b91c1c; font-weight:600; }
      .doc-item {
        font-size:.86rem; color:#374151; padding:3px 0; border-bottom:1px dashed #eef0f3;
      }
      /* Buttons */
      .stButton > button {
        border-radius:10px; border:1px solid #e5e7eb; font-weight:600;
        transition:all .15s ease;
      }
      .stButton > button:hover { border-color:#7c3aed; color:#6d28d9; transform:translateY(-1px); }
      /* Footer */
      .footer {
        text-align:center; color:#6b7280; font-size:.85rem; margin-top:26px;
        padding-top:14px; border-top:1px solid #eef0f3;
      }
      .footer a { color:#6d28d9; text-decoration:none; font-weight:600; }
      /* Never allow horizontal overflow on small screens */
      .hero, .side-card { max-width:100%; box-sizing:border-box; overflow-wrap:anywhere; }
      /* Mobile tweaks */
      @media (max-width: 640px) {
        .hero { padding:20px 18px; border-radius:16px; }
        .hero h1 { font-size:1.5rem; }
        .hero p  { font-size:.92rem; }
        .pill { font-size:.72rem; padding:4px 9px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing the knowledge base…")
def _prepare_index():
    """Build the vector index once per server session if it isn't there yet."""
    return ingest.ensure_index()


provider = config.LLM_PROVIDER
model = {
    "groq": config.GROQ_MODEL,
    "openai": config.OPENAI_MODEL,
    "ollama": config.OLLAMA_MODEL,
}.get(provider, "?")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown(
        f"""
        <div class="side-card">
          <p class="side-title">🔎 Research Assistant</p>
          <p class="side-sub">Agentic RAG · documents + web + calculator</p>
          <p class="side-sub" style="margin-top:8px;">Created by <b>{CREATOR}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    key_ok = (
        (provider == "groq" and os.environ.get("GROQ_API_KEY"))
        or (provider == "openai" and os.environ.get("OPENAI_API_KEY"))
        or (provider == "ollama")
    )

    try:
        n_chunks = _prepare_index()
        index_line = f'<span class="status-ok">✅ {n_chunks} chunks indexed</span>'
    except Exception as exc:  # noqa: BLE001
        index_line = f'<span class="status-bad">Index error: {exc}</span>'

    key_line = (
        '<span class="status-ok">✅ API key detected</span>'
        if key_ok
        else '<span class="status-bad">⚠️ Missing API key</span>'
    )

    st.markdown(
        f"""
        <div class="side-card">
          <p class="side-sub"><b>Status</b></p>
          <div class="doc-item">{key_line}</div>
          <div class="doc-item">{index_line}</div>
          <div class="doc-item">LLM: <code>{provider}</code> · <code>{model}</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    docs_html = ""
    if os.path.isdir(config.DOCS_DIR):
        for name in sorted(os.listdir(config.DOCS_DIR)):
            if os.path.splitext(name)[1].lower() in config.SUPPORTED_EXTENSIONS:
                docs_html += f'<div class="doc-item">📄 {name}</div>'
    st.markdown(
        f'<div class="side-card"><p class="side-sub"><b>Knowledge base</b></p>{docs_html}</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔁 Rebuild", use_container_width=True):
            ingest.build_index(force=True)
            _prepare_index.clear()
            st.rerun()
    with col_b:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown(
        f'<p class="side-sub" style="text-align:center;margin-top:10px;">'
        f'Built with LangGraph · Chroma · Groq<br/>© {CREATOR}</p>',
        unsafe_allow_html=True,
    )


# ---------------- Hero header ----------------
st.markdown(
    f"""
    <div class="hero">
      <h1>🔎 AI Research Assistant</h1>
      <p>An agentic RAG assistant that reads your documents, searches the web, and does math —
         and decides which tool to use for every question.</p>
      <div class="pills">
        <span class="pill">📄 Document RAG</span>
        <span class="pill">🌐 Web search</span>
        <span class="pill">🧮 Calculator</span>
        <span class="pill">🤖 LangGraph agent</span>
      </div>
      <div class="byline">✦ Built by {CREATOR}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Chat state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting" not in st.session_state:
    st.session_state.awaiting = False

EXAMPLES = [
    "What is the main idea behind the attention mechanism?",
    "How does retrieval-augmented generation reduce hallucination?",
    "What problem does LoRA solve in fine-tuning?",
    "What is the latest stable version of Python?",
]

# Show example prompts only on a fresh chat.
pending = None
if not st.session_state.messages:
    st.markdown("#### 💡 Try one of these")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            pending = ex

# Render the existing conversation.
for role, content in st.session_state.messages:
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

# If the last message is an unanswered question, generate the answer now.
# This runs in its OWN rerun and keeps `awaiting` True until an answer is
# appended, so a mobile keyboard/resize rerun can't lose the response — it
# simply retries on the next run until it completes.
if st.session_state.awaiting and st.session_state.messages:
    question = st.session_state.messages[-1][1]
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                history = st.session_state.messages[:-1]
                answer = ask(question, history=history)
            except Exception as exc:  # noqa: BLE001
                answer = f"⚠️ Error: {exc}"
        st.markdown(answer)
    st.session_state.messages.append(("assistant", answer))
    st.session_state.awaiting = False

# Input (chat box or an example click). Ignore new input while answering.
typed = st.chat_input("Ask a question about the documents, the web, or math…")
user_prompt = typed or pending

if user_prompt and not st.session_state.awaiting:
    st.session_state.messages.append(("user", user_prompt))
    st.session_state.awaiting = True
    st.rerun()

# ---------------- Footer ----------------
st.markdown(
    f"""
    <div class="footer">
      Made with ❤️ by <b>{CREATOR}</b> ·
      <a href="https://github.com/aditya1609/ai-research-assistant" target="_blank">View source on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
