"""Look up LangSmith *model configurations* by name (Module 6).

A model configuration (Settings -> Model configurations) is a named,
workspace-level bundle of provider + model + parameters + the secret that holds
the API key. LangSmith features pick from that library instead of taking a raw
provider name:

- online / offline LLM-as-judge evaluators  -> `playground_settings_id`
- Insights "Thinking" model (clustering)     -> `cluster_model`
- Insights "Summarization" model             -> `summary_model`

Each configuration carries per-feature availability flags that are set in the
Feature Access table of the same settings page. The SDK has no wrapper for
these yet; they live at the `/playground-settings` REST endpoint.
"""

from __future__ import annotations

from typing import Optional

from langsmith import Client
from langsmith import utils as ls_utils

# feature name used in this repo -> flag on the configuration object
FEATURE_FLAGS = {
    "evaluators": "available_in_evaluators",
    "insights_thinking": "available_in_insights_heavy",
    "insights_summarization": "available_in_insights_light",
    "playground": "available_in_playground",
}


def list_model_configurations(client: Client) -> list[dict]:
    """Return every model configuration in the workspace."""
    response = client.request_with_retries("GET", "/playground-settings")
    ls_utils.raise_for_status_with_text(response)
    data = response.json()
    return data["items"] if isinstance(data, dict) else data


def get_model_configuration(client: Client, name: str, *, feature: Optional[str] = None) -> dict:
    """Return the configuration named `name`.

    With `feature` set (one of `FEATURE_FLAGS`), also require that the
    configuration is enabled for that feature and explain how to fix it if not.
    """
    configs = list_model_configurations(client)
    match = next((c for c in configs if c.get("name") == name), None)
    if match is None:
        names = sorted(c.get("name") or "(unnamed)" for c in configs)
        raise KeyError(f"No model configuration named {name!r}. Available: {names}")
    if feature is not None:
        flag = FEATURE_FLAGS[feature]
        if not match.get(flag):
            raise ValueError(
                f"Model configuration {name!r} is not enabled for {feature} ({flag} is false). "
                "Turn it on under Settings -> Model configurations -> Feature Access."
            )
    return match


def _model_label(config: dict) -> str:
    settings = config.get("settings") or {}
    for key in ("model", "model_name", "deployment_name"):
        if settings.get(key):
            return str(settings[key])
    return "?"


def print_model_configurations(client: Client) -> None:
    """Print name, model, and feature availability for every configuration."""
    configs = list_model_configurations(client)
    if not configs:
        print("No model configurations yet. Create one under Settings -> Model configurations.")
        return
    print(f"{'name':32} {'model':28} {'evaluators':>10} {'insights:thinking':>18} {'insights:summary':>17}")
    for c in sorted(configs, key=lambda c: c.get("name") or ""):
        flags = [c.get(FEATURE_FLAGS[f]) for f in ("evaluators", "insights_thinking", "insights_summarization")]
        print(
            f"{(c.get('name') or '(unnamed)')[:32]:32} {_model_label(c)[:28]:28} "
            f"{'yes' if flags[0] else '-':>10} {'yes' if flags[1] else '-':>18} {'yes' if flags[2] else '-':>17}"
        )
