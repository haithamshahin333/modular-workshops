# Modular Workshops

This repository contains hands-on tutorials for learning LangChain, LangGraph, and Deep Agents.

This is a condensed version of LangChain Academy, intended to be run in a session with a LangChain engineer. If you're interested in going into more depth, or working through tutorials on your own, check out [LangChain Academy](https://academy.langchain.com/courses/intro-to-langgraph)! LangChain Academy has helpful pre-recorded videos from our LangChain engineers.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment variables
cp .env.example .env
# Edit .env and fill in your keys
```

| Key | Required for | Get one |
|-----|--------------|---------|
| `OPENAI_API_KEY` | Modules 1-4 (default model) | <https://platform.openai.com> |
| `LANGSMITH_API_KEY` | Modules 3 & 4 (recommended for all) | <https://smith.langchain.com> |
| `TAVILY_API_KEY` | Modules 2 & 3 (web search tool) | <https://tavily.com> |

```bash
# 3. Start Jupyter
uv run jupyter notebook
```

Open whichever module(s) your recipe calls for.

## Switching Models

All modules import `model` from `utils/models.py`. Change one line there to swap providers — no notebook edits required.

```python
# utils/models.py

# OpenAI (default)
model = init_chat_model("openai:gpt-4.1-mini")

# Anthropic
# model = init_chat_model("anthropic:claude-sonnet-4-5")

# Azure OpenAI
# from langchain_openai import AzureChatOpenAI
# model = AzureChatOpenAI(azure_deployment="gpt-4.1-mini", streaming=True)

# AWS Bedrock
# from langchain_aws import ChatBedrockConverse
# model = ChatBedrockConverse(provider="anthropic", model_id="...")
```

## Deploy (Module 3)

Module 3 deploys the agent at `agents/deep_agent/` to LangSmith via the `langgraph` CLI (installed by `uv sync`). The deploy config is `langgraph.json` at the workshop root.

Your `LANGSMITH_API_KEY` must have deployment permissions (use a `lsv2_sk_...` service key).

## Project Structure

```
modular-workshops/
├── README.md                       (this file — recipes + setup)
├── pyproject.toml                  (shared dependencies)
├── .env.example
├── langgraph.json                  (registers agents/deep_agent for langgraph dev)
├── utils/
├── agents/
│   ├── research_agent.py           (shared agent factory — Module 2 references, Module 4 imports for eval)
│   └── deep_agent/                 (deployable agent for Module 3)
│       ├── agent.py
│       ├── AGENTS.md
│       └── skills/
│           ├── linkedin-post/SKILL.md
│           └── twitter-post/SKILL.md
├── images/                         (diagrams used by the notebooks)
└── modules/
    ├── 01_langgraph.ipynb          (Module 1)
    ├── 02_deep_agents.ipynb        (Module 2)
    ├── 03_deploy.ipynb             (Module 3)
    └── 04_langsmith.ipynb          (Module 4)
```

## Common Issues

**`langgraph deploy` fails with 403 / permission denied**
Your API key is a personal token. Generate a service key (`lsv2_sk_...`) in LangSmith settings.

**Notebook can't find `utils` / `agents`**
Each module's setup cell prepends `project_root` (the workshop root) to `sys.path`. If you moved a notebook, update the `Path().resolve().parent` line to point at the workshop root.
