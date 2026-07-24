"""The agent: a tool-calling ReAct agent that works with Groq, OpenAI, or Ollama.

The LLM provider is chosen via config.LLM_PROVIDER (default: groq).
"""

import os
from functools import lru_cache

from langgraph.prebuilt import create_react_agent

import config
from src.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a Personal Research Assistant.

You have three tools:
1. search_documents  - search the user's private indexed documents. Use this FIRST
   for anything that could be answered from the internal knowledge base.
2. web_search        - search the public web for current or general information
   that is not in the documents.
3. calculator        - do exact arithmetic.

Guidelines:
- Prefer search_documents for company/product/internal questions.
- Use web_search only when the documents don't have the answer or the question is
  about current/general world knowledge.
- Always use the calculator for math instead of doing it in your head.
- Cite the source file name when you answer from documents.
- If you don't find an answer, say so honestly. Be concise and clear.
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
        # format for tool-calling to work reliably on Groq.
        model = config.GROQ_MODEL
        if any(tag in model for tag in ("gpt-oss", "qwen", "deepseek")):
            extra["reasoning_format"] = "parsed"

        return ChatGroq(
            model=model,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
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

    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content


if __name__ == "__main__":
    print(ask("How many PTO days do full-time employees get, and what is that per month?"))
