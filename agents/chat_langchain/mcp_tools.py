"""Shared MCP tool loader for the Chat LangChain agents (Module 6).

Both agents in this package talk to the two public LangChain documentation MCP
servers (see https://docs.langchain.com/use-these-docs):

- ``docs``      https://docs.langchain.com/mcp       — guides, how-tos, concepts, product docs
- ``reference`` https://reference.langchain.com/mcp  — API reference: classes, methods, signatures

Both use streamable-HTTP transport and need no API key. ``MultiServerMCPClient``
is stateless: every tool call opens a short-lived MCP session, so there is
nothing to tear down and the same client is safe to share across graphs.

MCP tools are async-only (``ainvoke``). That is why every node and agent in this
package is async, and why the notebook uses top-level ``await`` and
``client.aevaluate`` instead of the sync APIs used in Module 4.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

DOCS_MCP_URL = "https://docs.langchain.com/mcp"
REFERENCE_MCP_URL = "https://reference.langchain.com/mcp"

MCP_SERVERS = {
    "docs": {"transport": "http", "url": DOCS_MCP_URL},
    "reference": {"transport": "http", "url": REFERENCE_MCP_URL},
}

# Tool names exactly as the servers expose them.
DOCS_SEARCH_TOOL = "search_docs_by_lang_chain"
DOCS_FS_TOOL = "query_docs_filesystem_docs_by_lang_chain"
REFERENCE_SEARCH_TOOL = "search_api"
REFERENCE_SYMBOL_TOOL = "get_symbol"

DOCS_TOOLS = [DOCS_SEARCH_TOOL, DOCS_FS_TOOL]
REFERENCE_TOOLS = [REFERENCE_SEARCH_TOOL, REFERENCE_SYMBOL_TOOL]

# `submit_feedback` files a report with the LangChain docs team. An agent must
# never be able to call it on a user's behalf, so it is dropped when loading.
EXCLUDED_TOOLS = {"submit_feedback"}

_client = MultiServerMCPClient(MCP_SERVERS)
_tools: dict[str, BaseTool] | None = None


async def get_mcp_tools(refresh: bool = False) -> dict[str, BaseTool]:
    """Return ``{tool_name: tool}`` for both servers, cached after the first call.

    The first call lists tools over the network (one round-trip per server);
    later calls return the cached dict. Pass ``refresh=True`` to re-list.
    """
    global _tools
    if _tools is None or refresh:
        loaded = await _client.get_tools()
        _tools = {t.name: t for t in loaded if t.name not in EXCLUDED_TOOLS}
    return _tools


def select_tools(tools: dict[str, BaseTool], names: list[str]) -> list[BaseTool]:
    """Pick tools by name, failing loudly if a server renamed one."""
    missing = [n for n in names if n not in tools]
    if missing:
        raise KeyError(
            f"MCP tools not found: {missing}. Available: {sorted(tools)}. "
            "The docs servers may have renamed a tool; update mcp_tools.py."
        )
    return [tools[n] for n in names]


def tool_result_text(result: Any) -> str:
    """Normalize an MCP tool result to plain text.

    ``langchain-mcp-adapters`` returns either a string or a list of content
    blocks (``{"type": "text", "text": ...}``), depending on the server.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for block in result:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(getattr(block, "text", block)))
        return "\n".join(p for p in parts if p)
    return str(result)


_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


def extract_urls(text: str) -> list[str]:
    """Return the unique URLs in ``text``, in order of first appearance."""
    seen: list[str] = []
    for url in _URL_RE.findall(text):
        url = url.rstrip(".,;:")
        if url not in seen:
            seen.append(url)
    return seen
