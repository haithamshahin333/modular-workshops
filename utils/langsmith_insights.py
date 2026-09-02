"""Create and schedule LangSmith Insights reports from code (Module 6).

Insights samples traces from a tracing project, summarizes each one, clusters
the summaries into categories, and writes an executive summary. The UI drives
it from **+ New -> New Insights Report**; the SDK's `client.generate_insights`
covers only chat histories uploaded from *outside* LangSmith. To run Insights
on an existing project we call the same REST endpoint the UI uses:

    POST /sessions/{project_id}/insights            -- run a report now
    POST /sessions/{project_id}/insights/configs    -- save a config (+ cron schedule)

Requires a Plus/Enterprise plan and a model provider secret (`OPENAI_API_KEY`)
in Workspace settings -> Secrets. Reports take up to ~30 minutes.

Insights uses two models: a "Thinking" model for clustering and a
"Summarization" model for per-trace summaries. By default it uses the
provider's defaults (`model="openai"|"anthropic"`); pass
`thinking_model_config` / `summarization_model_config` (names from Settings ->
Model configurations, enabled for Insights) to pin specific configurations.
Those are sent as `cluster_model` / `summary_model`.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from langsmith import Client
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils

from utils.langsmith_model_configs import get_model_configuration

# Summary prompt for a chat project. Mustache variables available:
#   {{run.inputs}} {{run.outputs}} {{run.error}} {{run.feedback}} {{run.feedback.<key>}}
#   {{all_thread_messages}} (full conversation when the project has threads)
DEFAULT_SUMMARY_PROMPT = """Summarize this conversation between a user and a documentation assistant.

Conversation:
{{all_thread_messages}}

Evaluator feedback on the last turn (JSON):
{{run.feedback}}

In 2-3 sentences state: what the user was trying to learn (topic + task), whether the assistant answered it with citations, and any sign the user had to repeat or correct themselves."""


def create_insights_report(
    client: Client,
    project_name: str,
    *,
    name: str,
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
    attribute_schemas: Optional[dict[str, dict[str, Any]]] = None,
    last_n_hours: int = 24,
    sample: int = 200,
    filter: Optional[str] = None,
    model: Literal["openai", "anthropic"] = "openai",
    thinking_model_config: Optional[str] = None,
    summarization_model_config: Optional[str] = None,
) -> ls_schemas.InsightsReport:
    """Kick off an Insights report over a tracing project; returns a report with a `.link`.

    `attribute_schemas` adds per-trace attributes the summarizer extracts, e.g.
    `{"answered_with_citation": {"type": "boolean", "description": "..."}}`.
    Attributes steer clustering and show up as aggregates per category.

    `thinking_model_config` / `summarization_model_config` are model
    configuration *names*; each must be enabled for the matching Insights
    feature or a `ValueError` explains what to switch on.
    """
    project = client.read_project(project_name=project_name)
    config = _job_config(
        client,
        name=name,
        summary_prompt=summary_prompt,
        attribute_schemas=attribute_schemas,
        last_n_hours=last_n_hours,
        sample=sample,
        filter=filter,
        model=model,
        thinking_model_config=thinking_model_config,
        summarization_model_config=summarization_model_config,
    )
    response = client.request_with_retries("POST", f"/sessions/{project.id}/insights", json=config)
    ls_utils.raise_for_status_with_text(response)
    body = response.json()
    return ls_schemas.InsightsReport(
        id=body["id"],
        name=body["name"],
        status=body["status"],
        error=body.get("error"),
        project_id=project.id,
        tenant_id=client._get_tenant_id(),
        host_url=client._host_url,
    )


def save_insights_config(
    client: Client,
    project_name: str,
    *,
    name: str,
    schedule_cron: Optional[str] = None,
    description: Optional[str] = None,
    **job_kwargs: Any,
) -> dict:
    """Save a reusable Insights config, optionally on a cron schedule (UTC).

    `job_kwargs` are the same as `create_insights_report` (summary_prompt,
    attribute_schemas, last_n_hours, sample, filter, model,
    thinking_model_config, summarization_model_config). Scheduled runs
    recompute the time window each time, so `last_n_hours=168` + weekly cron
    yields comparable week-over-week reports.
    """
    project = client.read_project(project_name=project_name)
    body = {
        "name": name,
        "description": description,
        "config": _job_config(client, name=name, **job_kwargs),
        "schedule_cron": schedule_cron,
    }
    response = client.request_with_retries("POST", f"/sessions/{project.id}/insights/configs", json=body)
    ls_utils.raise_for_status_with_text(response)
    return response.json()


def print_insights_summary(result: ls_schemas.InsightsReportResult) -> None:
    """Print the executive summary and top-level categories of a finished report."""
    print(f"{result.name}  [{result.status}]")
    if result.report and result.report.key_points:
        print("\nKey points:")
        for point in result.report.key_points:
            print(f"  - {point}")
    top = [c for c in result.clusters if getattr(c, "level", 0) == 0]
    if top:
        print("\nTop-level categories:")
        for c in sorted(top, key=lambda c: -(c.num_runs or 0)):
            print(f"  {c.num_runs:>4}  {c.name}: {(c.description or '')[:110]}")


def _job_config(
    client: Client,
    *,
    name: str,
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
    attribute_schemas: Optional[dict[str, dict[str, Any]]] = None,
    last_n_hours: int = 24,
    sample: int = 200,
    filter: Optional[str] = None,
    model: Literal["openai", "anthropic"] = "openai",
    thinking_model_config: Optional[str] = None,
    summarization_model_config: Optional[str] = None,
) -> dict:
    config: dict[str, Any] = {
        "name": name,
        "summary_prompt": summary_prompt,
        "last_n_hours": last_n_hours,
        "sample": sample,
        "model": model,
    }
    if attribute_schemas:
        config["attribute_schemas"] = attribute_schemas
    if filter:
        config["filter"] = filter
    # The API types these as plain strings. We send the configuration id, which is
    # what the UI's model picker selects; if the API rejects it, the fallback is the
    # model name from the configuration's `settings`.
    if thinking_model_config:
        config["cluster_model"] = get_model_configuration(
            client, thinking_model_config, feature="insights_thinking")["id"]
    if summarization_model_config:
        config["summary_model"] = get_model_configuration(
            client, summarization_model_config, feature="insights_summarization")["id"]
    return config
