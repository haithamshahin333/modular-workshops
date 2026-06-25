"""Setup helpers for the Module 5 Engine demo.

Resolve the deployed graph's URL and upsert the `engine-demo` assistant
(deep_agent + the deliberately broken `easy_search` tool + weak Context Hub
guidance). Kept out of the notebook so the cells stay short.
"""

from __future__ import annotations

import os
import re
import subprocess

ASSISTANT_NAME = "engine-demo"
GRAPH_ID = "deep_agent"

# deep_agent + the broken search tool and weak context that give Engine a
# clear, recurring failure to find. Interrupts off so seed runs finish unattended.
ENGINE_CONFIG = {"configurable": {
    "search_tool": "easy",
    "interrupts": False,
    "context_repo": "engine-demo-context",
}}


def deployment_url(deployment_name: str) -> str:
    """Return the deployment URL from $LANGSMITH_DEPLOYMENT_URL or the langgraph CLI."""
    url = os.environ.get("LANGSMITH_DEPLOYMENT_URL")
    if url:
        return url
    out = subprocess.run(
        ["langgraph", "deploy", "list", "--name-contains", deployment_name],
        capture_output=True, text=True,
    ).stdout
    match = re.search(r"https://\S+\.langgraph\.app", out)
    if not match:
        raise RuntimeError(f"No deployment named {deployment_name!r}. Deploy it (Module 3) first.")
    return match.group(0)


def upsert_engine_assistant(sdk) -> str:
    """Create or update the engine-demo assistant; return its id."""
    match = next(
        (a for a in sdk.assistants.search(limit=100)
         if a.get("name") == ASSISTANT_NAME and a.get("graph_id") == GRAPH_ID),
        None,
    )
    if match is None:
        assistant = sdk.assistants.create(
            GRAPH_ID, config=ENGINE_CONFIG, name=ASSISTANT_NAME, if_exists="do_nothing")
    else:
        assistant = sdk.assistants.update(match["assistant_id"], config=ENGINE_CONFIG)
    return assistant["assistant_id"]
