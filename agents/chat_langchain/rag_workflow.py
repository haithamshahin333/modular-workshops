"""LangGraph RAG workflow over the LangChain docs (Module 6, graph id ``rag_workflow``).

A deliberately *deterministic* pipeline, the counterpart to the Deep Agent in
``deep_agent.py``: the model never decides *whether* to search. Every turn runs

    plan_query -> retrieve_docs and/or retrieve_reference -> generate

``plan_query`` rewrites the latest message into a standalone search query and
picks a route; the retrieval nodes call the MCP docs tools directly (each call
shows up as a tool run in the trace); ``generate`` answers only from the
retrieved context and cites URLs.

State extends ``MessagesState`` so the graph speaks the same ``messages``
contract as the Deep Agent (Studio chat mode, the mini UI, thread-level
evaluators, and one shared eval dataset all rely on that).
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from agents.chat_langchain.mcp_tools import (
    DOCS_SEARCH_TOOL,
    REFERENCE_SEARCH_TOOL,
    extract_urls,
    get_mcp_tools,
    tool_result_text,
)
from utils.models import model


class RAGState(MessagesState):
    query: str            # standalone search query for the latest turn
    route: str            # "docs" | "reference" | "both"
    context: str          # retrieved text handed to `generate`
    sources: list[str]    # URLs found in the retrieved text
    tool_calls: list[str]  # MCP tools invoked this turn (for trajectory evals)


class QueryPlan(BaseModel):
    """Retrieval plan for the user's latest message."""

    query: str = Field(
        description=(
            "One standalone search query that captures the user's latest question, "
            "resolving pronouns and follow-ups using the conversation."
        )
    )
    route: Literal["docs", "reference", "both"] = Field(
        description=(
            "'docs' for concepts, how-tos, and product questions; "
            "'reference' for exact class/function signatures, parameters, or return types; "
            "'both' when the question needs a how-to AND exact API details."
        )
    )


PLANNER_PROMPT = (
    "You plan retrieval for an assistant that answers questions about LangChain, "
    "LangGraph, Deep Agents, and LangSmith from the official documentation. "
    "Given the conversation, write one standalone search query for the user's "
    "latest message and choose where to search."
)

ANSWER_PROMPT = """You are Chat LangChain, an assistant for questions about LangChain, LangGraph, Deep Agents, and LangSmith.

Answer the user's latest message using ONLY the retrieved context below.
- Be direct and specific. Include a short code snippet when the context contains one.
- Cite the documentation URL(s) you relied on inline, as plain URLs.
- If the context does not answer the question, say so plainly and suggest what to search next.
- Never invent APIs, parameters, or URLs.

Retrieved context:
{context}
"""

planner = model.with_structured_output(QueryPlan)


async def plan_query(state: RAGState) -> dict:
    plan = await planner.ainvoke([SystemMessage(content=PLANNER_PROMPT), *state["messages"]])
    # Reset per-turn fields so a multi-turn thread doesn't accumulate old context.
    return {"query": plan.query, "route": plan.route, "context": "", "sources": [], "tool_calls": []}


async def retrieve_docs(state: RAGState) -> dict:
    tools = await get_mcp_tools()
    text = tool_result_text(await tools[DOCS_SEARCH_TOOL].ainvoke({"query": state["query"]}))
    return {
        "context": state["context"] + "\n\n## Documentation search results\n" + text,
        "sources": state["sources"] + extract_urls(text),
        "tool_calls": state["tool_calls"] + [DOCS_SEARCH_TOOL],
    }


async def retrieve_reference(state: RAGState) -> dict:
    tools = await get_mcp_tools()
    text = tool_result_text(
        await tools[REFERENCE_SEARCH_TOOL].ainvoke(
            {"query": state["query"], "language": "python", "limit": 3}
        )
    )
    return {
        "context": state["context"] + "\n\n## API reference search results\n" + text,
        "sources": state["sources"] + extract_urls(text),
        "tool_calls": state["tool_calls"] + [REFERENCE_SEARCH_TOOL],
    }


async def generate(state: RAGState) -> dict:
    system = ANSWER_PROMPT.format(context=state["context"].strip() or "(nothing retrieved)")
    response = await model.ainvoke([SystemMessage(content=system), *state["messages"]])
    return {"messages": [response]}


def route_after_plan(state: RAGState) -> Literal["retrieve_docs", "retrieve_reference"]:
    return "retrieve_reference" if state["route"] == "reference" else "retrieve_docs"


def route_after_docs(state: RAGState) -> Literal["retrieve_reference", "generate"]:
    return "retrieve_reference" if state["route"] == "both" else "generate"


def build_graph(checkpointer=None):
    """Compile the workflow. Pass a checkpointer for local multi-turn use; the
    deployment injects its own, so the module-level ``graph`` has none."""
    builder = StateGraph(RAGState)
    builder.add_node("plan_query", plan_query)
    builder.add_node("retrieve_docs", retrieve_docs)
    builder.add_node("retrieve_reference", retrieve_reference)
    builder.add_node("generate", generate)

    builder.add_edge(START, "plan_query")
    builder.add_conditional_edges("plan_query", route_after_plan, ["retrieve_docs", "retrieve_reference"])
    builder.add_conditional_edges("retrieve_docs", route_after_docs, ["retrieve_reference", "generate"])
    builder.add_edge("retrieve_reference", "generate")
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
