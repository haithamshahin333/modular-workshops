"""Centralized model initialization.

The notebooks all import `model` from here, so swapping providers
(OpenAI / Anthropic / Azure / Bedrock) only requires editing this file.

Each route below is two lines: `MODEL_NAME` (which model) and `MODEL_KWARGS` (how to
reach it). Uncomment exactly one. `model` is the shared default every notebook imports;
`make_model("<other-model>")` builds a *second* model on the same route — same provider,
same gateway, same credentials — which is what Module 6 §5.6 uses to re-run an experiment
against a different model without turning a model swap into a gateway bypass.

The default is OpenAI, direct, on the Responses API. Module 3 §1.4 walks
through introducing the LangSmith LLM Gateway as the production-ready
alternative — to make the switch, comment out the default below and uncomment
the gateway block.
"""

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env", override=True)

from langchain.chat_models import init_chat_model

# --- Default: OpenAI, direct ---
# MODEL_NAME = "gpt-5.6-terra"
# MODEL_KWARGS = {"model_provider": "openai", "use_responses_api": True}

# --- OpenAI via the LangSmith LLM Gateway (Module 3 §1.4) ---
# Routes every model call through the LangSmith Gateway so that workspace
# policies (PII / secrets / allow-lists / cost caps) are enforced.
MODEL_NAME = "gpt-5.4-mini"
MODEL_KWARGS = {
    "model_provider": "openai",
    "base_url": "https://gateway.smith.langchain.com/openai",
    "use_responses_api": True,
    "api_key": os.environ["LANGSMITH_API_KEY_GATEWAY"],
}

# --- Anthropic ---
# MODEL_NAME = "claude-sonnet-5"
# MODEL_KWARGS = {"model_provider": "anthropic"}


def make_model(name: str | None = None, **overrides):
    """Build a chat model on whichever route is uncommented above.

    `make_model()` is the shared default; `make_model("gpt-5.6-terra")` keeps the same
    provider, endpoint, and credentials and swaps only the model.

    `overrides` is for per-model tuning (temperature, max_tokens, ...). Passing
    `base_url` or `api_key` here would route around the gateway configured above, so
    change the route in this file rather than at the call site.
    """
    return init_chat_model(model=name or MODEL_NAME, **{**MODEL_KWARGS, **overrides})


# --- Azure OpenAI ---
# Azure routes on the *deployment* name, which `init_chat_model` does not set, so this
# route replaces the factory rather than MODEL_NAME / MODEL_KWARGS.
# from langchain_openai import AzureChatOpenAI
# def make_model(name: str | None = None, **overrides):
#     return AzureChatOpenAI(
#         azure_deployment=name or "gpt-5.6-terra", streaming=True, **overrides)

# --- AWS Bedrock ---
# from langchain_aws import ChatBedrockConverse
# def make_model(name: str | None = None, **overrides):
#     return ChatBedrockConverse(
#         provider="anthropic",
#         model_id=name or "anthropic.claude-sonnet-4-20250514-v1:0",
#         **overrides,
#     )


model = make_model()
