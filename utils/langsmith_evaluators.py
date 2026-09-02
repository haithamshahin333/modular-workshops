"""Helpers for managing LangSmith *evaluators* with the SDK (Module 6).

Since `langsmith>=0.9.8` evaluators are workspace-level resources managed via
`client.evaluators` (create / list / update / delete). They show up in the
**Evaluators** table in the UI and can be attached to any tracing project
(online evaluation) or dataset (offline evaluation).

The pieces, in the order the notebook uses them:

1. `push_judge_prompt`  -- an LLM-as-judge evaluator references a *structured*
   prompt in the Prompt Hub. We push `StructuredPrompt | ChatOpenAI` so the hub
   commit carries a default judge model; LangSmith resolves the model's API key
   from the workspace secret `OPENAI_API_KEY` at run time.
2. `upsert_evaluator`   -- create (or update, if the name exists) an `llm` or
   `code` evaluator. Idempotent so the notebook can be re-run. Pass
   `model_configuration="<name>"` to run the judge on a named model
   configuration (Settings -> Model configurations) instead of the hub model.
3. `attach_evaluator`   -- attach it to a tracing project via a run rule
   (`utils.langsmith_rules.create_run_rule`). `group_by="thread_id"` turns it
   into a thread-level (multi-turn) evaluator.

`run_code_evaluator_locally` lets you test the sandboxed `perform_eval` source
against a real run dict before uploading it.

Note: `client.evaluators.*` methods are async -- `await` them (Jupyter supports
top-level `await`).
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union
from uuid import UUID

from langchain_core.prompts.structured import StructuredPrompt
from langchain_openai import ChatOpenAI
from langsmith import Client
from pydantic import BaseModel

from utils.langsmith_model_configs import get_model_configuration
from utils.langsmith_rules import create_run_rule

# Same judge model Module 4 uses for its online evaluator.
JUDGE_MODEL_NAME = "gpt-5.6-luna"


def push_judge_prompt(
    client: Client,
    name: str,
    *,
    system: str,
    human: str,
    schema: type[BaseModel],
    model_name: str = JUDGE_MODEL_NAME,
) -> dict:
    """Push a structured judge prompt (+ model) to the Prompt Hub.

    Returns ``{"handle": <prompt_repo_handle>, "commit": <commit hash>, "url": <hub url>}``;
    pass ``handle`` and ``commit`` to ``upsert_evaluator`` so the evaluator pins the
    exact commit you just pushed.

    `system` / `human` are prompt templates. Use single-brace variables
    (`{question}`, `{response}`, or `{all_messages}` for thread-level judges);
    `variable_mapping` on the evaluator maps run fields onto those names.
    `schema` is a Pydantic model describing the judge's structured output --
    LangSmith writes one feedback score per field (booleans/numbers) and treats
    a `reasoning`/`comment` string as the feedback comment.
    """
    prompt = StructuredPrompt.from_messages_and_schema(
        [("system", system), ("human", human)],
        schema=schema.model_json_schema(),
    )
    # The API key is serialized as a *reference* to the workspace secret
    # OPENAI_API_KEY, never as a value.
    judge = ChatOpenAI(model=model_name)
    url = client.push_prompt(name, object=prompt | judge)
    # Hub URLs end in /<prompt-name>/<commit-hash>[?query]
    commit = url.split("?")[0].rstrip("/").split("/")[-1]
    return {"handle": name, "commit": commit, "url": url}


async def upsert_evaluator(
    client: Client,
    name: str,
    *,
    type: Literal["llm", "code"],
    llm_evaluator: Optional[dict[str, Any]] = None,
    code_evaluator: Optional[dict[str, Any]] = None,
    model_configuration: Optional[str] = None,
) -> str:
    """Create a workspace evaluator, or update the one with this exact name. Returns its id.

    - `type="llm"`: `llm_evaluator={"prompt_repo_handle", "commit_hash_or_tag",
      "variable_mapping": {prompt_var: "inputs.<path>" | "outputs.<path>"}}`.
      `model_configuration` is the *name* of a model configuration enabled for
      Evaluators; it is resolved to `playground_settings_id` and takes precedence
      over any model stored in the hub commit.
    - `type="code"`: `code_evaluator={"code": "<python source>", "language": "python"}`
      where the source defines `perform_eval(run, example)` and returns
      `{feedback_key: score, ...}`.
    """
    if type == "llm" and model_configuration is not None:
        config = get_model_configuration(client, model_configuration, feature="evaluators")
        llm_evaluator = {**(llm_evaluator or {}), "playground_settings_id": config["id"]}
    kwargs = {"llm_evaluator": llm_evaluator} if type == "llm" else {"code_evaluator": code_evaluator}

    page = await client.evaluators.list(name_contains=name, limit=50)
    existing = next((e for e in page.evaluators if e.name == name), None)
    if existing is not None:
        await client.evaluators.update(existing.id, name=name, **kwargs)
        return existing.id

    created = await client.evaluators.create(name=name, type=type, **kwargs)
    return created.evaluator.id


def attach_evaluator(
    client: Client,
    evaluator_id: Union[str, UUID],
    *,
    project_name: str,
    display_name: str,
    filter: str = "eq(is_root, true)",
    sampling_rate: float = 1.0,
    group_by: Optional[str] = None,
    include_extended_stats: bool = False,
) -> dict:
    """Attach an evaluator to a tracing project as an online evaluator.

    Returns `{"id", "url", "payload"}` from `create_run_rule`. Re-running with
    the same `display_name` replaces the previous rule.
    """
    return create_run_rule(
        client,
        project_name=project_name,
        display_name=display_name,
        filter=filter,
        sampling_rate=sampling_rate,
        evaluator_id=evaluator_id,
        group_by=group_by,
        include_extended_stats=include_extended_stats,
    )


def run_code_evaluator_locally(code: str, run: dict, example: Optional[dict] = None) -> dict:
    """Execute a code evaluator's source against a run dict, the way LangSmith's sandbox does.

    The sandbox has no network access and only the standard library plus
    numpy / pandas / jsonschema / scipy / scikit-learn -- keep `code` within that.
    """
    namespace: dict[str, Any] = {}
    exec(code, namespace)
    return namespace["perform_eval"](run, example)


def run_to_dict(run) -> dict:
    """Turn a `langsmith.schemas.Run` into the plain dict shape code evaluators receive."""
    data = run.model_dump(mode="json") if hasattr(run, "model_dump") else dict(run)
    return {k: v for k, v in data.items() if not k.startswith("_")}
