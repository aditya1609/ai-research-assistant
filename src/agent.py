"""The agent: a tool-calling ReAct agent that works with Groq, OpenAI, or Ollama.

The LLM provider is chosen via config.LLM_PROVIDER (default: groq).
"""

import os
from functools import lru_cache

from langgraph.prebuilt import create_react_agent

import config
from src.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a Personal Research Assistant that answers questions
using an indexed library of documents (research papers and notes), the web, and
a calculator.

You have three tools:
1. search_documents  - semantic search over the user's indexed documents.
2. web_search        - search the public web.
3. calculator        - exact arithmetic.

TOOL-SELECTION RULES (follow strictly):
- ALWAYS call `search_documents` FIRST for any question that could plausibly be
  covered by the documents. The library is about AI / machine learning / research
  topics (e.g. transformers, attention, embeddings, retrieval-augmented
  generation, fine-tuning, LoRA), so treat ANY technical or conceptual question
  as a documents question and search the documents before anything else.
- Only call `web_search` if `search_documents` returns nothing relevant, OR the
  question is clearly about current events / real-world facts not in the papers.
  If you do fall back to the web, say so explicitly in your answer.
- ALWAYS use `calculator` for arithmetic instead of computing in your head.

ANSWER STYLE:
- Write a thorough, well-structured answer: aim for 5-10 sentences (or a few short
  paragraphs). Explain the concept clearly, include the key details, the intuition,
  and an example or the "why it matters" where helpful.
- When you used the documents, ground the answer in the retrieved text and end with
  a "Source:" line naming the document file(s) you used.
- If the documents don't contain the answer, be honest about it and clearly label
  any information that came from the web.
"""


def _build_llm():
    provider = config.LLM_PROVIDER

    if provider == "groq":
        from langchain_groq import ChatGroq

        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and add it to your environment "
                "or Streamlit/HF secrets."
            )
        # Corporate networks often do SSL inspection with a private CA that
        # Python doesn't trust, which surfaces as a "Connection error".
        # Set SSL_VERIFY=false to tolerate that (fine for a corporate proxy).
        extra = {}
        if os.environ.get("SSL_VERIFY", "true").lower() == "false":
            import httpx

            extra["http_client"] = httpx.Client(verify=False)
            extra["http_async_client"] = httpx.AsyncClient(verify=False)

        # Reasoning models (gpt-oss, qwen, deepseek) need a parsed reasoning
        # format for tool-calling to work reliably on Groq. We also keep the
        # reasoning effort low so responses stay fast.
        model = config.GROQ_MODEL
        if any(tag in model for tag in ("gpt-oss", "qwen", "deepseek")):
            extra["reasoning_format"] = "parsed"
            extra["reasoning_effort"] = "low"

        return ChatGroq(
            model=model,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            request_timeout=60,
            max_retries=2,
            **extra,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.TEMPERATURE,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


@lru_cache(maxsize=1)
def get_agent():
    llm = _build_llm()
    return create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)


def ask(question: str, history=None) -> str:
    """Ask the agent a question. `history` is an optional list of (role, content)."""
    agent = get_agent()

    messages = []
    for role, content in (history or []):
        messages.append((role, content))
    messages.append(("user", question))

    result = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 12},
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    print(ask("How many PTO days do full-time employees get, and what is that per month?"))
