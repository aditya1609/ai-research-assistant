"""Tools the agent can call: document search (RAG), web search, and a calculator."""

import ast
import operator as op

from langchain_core.tools import tool

import config
from src.store import get_vectorstore


@tool
def search_documents(query: str) -> str:
    """Search the user's private knowledge base (the indexed documents) for
    information relevant to the query. Use this FIRST for any question that could
    be answered by the internal documents (company policy, product, notes, etc.).
    """
    try:
        vs = get_vectorstore()
        results = vs.similarity_search(query, k=config.TOP_K)
    except Exception as exc:  # noqa: BLE001
        return f"Document search failed: {exc}"

    if not results:
        return "No relevant information found in the documents."

    blocks = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        cite = f"{source}" + (f" (p.{page})" if page else "")
        blocks.append(f"[{i}] Source: {cite}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


@tool
def web_search(query: str) -> str:
    """Search the public web for current or general information that is NOT in the
    internal documents (e.g. news, definitions, recent facts). Returns top results.
    """
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=config.TOP_K))
    except Exception as exc:  # noqa: BLE001
        return f"Web search failed: {exc}"

    if not hits:
        return "No web results found."

    blocks = []
    for i, h in enumerate(hits, 1):
        title = h.get("title", "")
        body = h.get("body", "")
        href = h.get("href", "")
        blocks.append(f"[{i}] {title}\n{body}\n{href}")
    return "\n\n".join(blocks)


# --- safe calculator (no eval of arbitrary code) ---
_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.FloorDiv: op.floordiv,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):  # numbers
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed.")
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (e.g. '1200 * 0.8 + 50').
    Supports + - * / // % ** and parentheses. Use this for any math.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as exc:  # noqa: BLE001
        return f"Could not evaluate '{expression}': {exc}"


ALL_TOOLS = [search_documents, web_search, calculator]
