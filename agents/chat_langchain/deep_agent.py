"""Deep Agent over the LangChain docs (Module 6, graph id ``chat_deep_agent``).

The *agentic* counterpart to ``rag_workflow.py``: the model decides when and
how often to search, can read whole doc pages, and delegates exact API
questions to an ``api-reference`` subagent that has the reference-server tools.

Exported as an async factory (``make_graph``) because the MCP tools have to be
loaded with ``await`` before ``create_deep_agent`` can bind them. The Agent
Server only injects arguments typed ``RunnableConfig`` / ``ServerRuntime`` into
factories, so the notebook uses ``make_local_agent`` to pass a checkpointer.
"""

from pathlib import Path

from deepagents import create_deep_agent
from langchain_core.runnables import RunnableConfig

from agents.chat_langchain.mcp_tools import DOCS_TOOLS, REFERENCE_TOOLS, get_mcp_tools, select_tools
from utils.models import model

AGENT_DIR = Path(__file__).resolve().parent

# The agent's identity lives in AGENTS.md (same convention as agents/deep_agent/).
# It is inlined as the system prompt so the deployment needs no filesystem backend.
SYSTEM_PROMPT = (AGENT_DIR / "AGENTS.md").read_text()

API_REFERENCE_SUBAGENT = {
    "name": "api-reference",
    "description": (
        "Looks up exact class/function signatures, parameters, defaults, and return "
        "types in the LangChain API reference. Give it one symbol or one precise "
        "question at a time."
    ),
    "system_prompt": (
        "You are an API reference specialist for the LangChain packages. Use "
        "search_api to find the right symbol, then get_symbol to read its full "
        "signature and parameters. Reply with the exact signature, the parameters "
        "that matter for the question, and the reference URL. Do not guess: if the "
        "symbol is not found, say so."
    ),
}


def build_agent(tools: dict, *, checkpointer=None):
    """Assemble the Deep Agent from already-loaded MCP tools."""
    return create_deep_agent(
        model=model,
        tools=select_tools(tools, DOCS_TOOLS),
        system_prompt=SYSTEM_PROMPT,
        subagents=[{**API_REFERENCE_SUBAGENT, "tools": select_tools(tools, REFERENCE_TOOLS)}],
        checkpointer=checkpointer,
    )


async def make_graph(config: RunnableConfig):
    """Factory the Agent Server calls (see langgraph.chat_langchain.json).
    The server injects `config`; it is unused here. The platform supplies the checkpointer."""
    return build_agent(await get_mcp_tools())


async def make_local_agent(checkpointer=None):
    """Notebook helper: same agent, with an in-memory checkpointer for multi-turn threads."""
    return build_agent(await get_mcp_tools(), checkpointer=checkpointer)
