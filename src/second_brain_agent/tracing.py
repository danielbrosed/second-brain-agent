"""Langfuse wiring, following its tracing best practices.

* One trace per unit of work: a question, or a captured decision (drafted, paused for a person,
  then written). A capture keeps one trace across the pause, even across processes, because its
  trace id is derived from the LangGraph thread id.
* The root observation carries a stable, verb-first name and a readable input and output (the
  question and the answer), not the raw graph state.
* Everything that leaves the process passes through `mask`, which redacts anything shaped like
  a credential: the same rule the vault linter enforces on the notes.
* Optional: without keys every helper here is a no-op and the agent runs untraced.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from . import __version__
from .vault import redact_secrets

HIDDEN = "langsmith:hidden"  # LangChain tag the Langfuse callback records at DEBUG level

# Framework internals that say nothing about what the agent did: LangGraph's start step and its
# conditional-edge routers (the functions in graph.py), and output parsers.
_NOISE = {"__start__", "entry", "after_check", "after_draft", "after_review", "RunnableLambda"}

_client = None


def enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def mask(*, data: Any, **_: Any) -> Any:
    if isinstance(data, str):
        return redact_secrets(data)
    if isinstance(data, dict):
        return {key: mask(data=value) for key, value in data.items()}
    if isinstance(data, list | tuple):
        return [mask(data=value) for value in data]
    return data


def should_export(span) -> bool:
    """Langfuse's default filter, minus framework noise and anything recorded at DEBUG level."""
    from langfuse import LangfuseOtelSpanAttributes, is_default_export_span

    if not is_default_export_span(span):
        return False
    name = span.name or ""
    if name in _NOISE or name.endswith("OutputParser"):
        return False
    return (span.attributes or {}).get(LangfuseOtelSpanAttributes.OBSERVATION_LEVEL) != "DEBUG"


def client():
    """The single Langfuse client; created before any handler so they all share its masking."""
    global _client
    if _client is None:
        from langfuse import Langfuse

        _client = Langfuse(
            mask=mask,
            should_export_span=should_export,
            environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development"),
            release=__version__,
        )
    return _client


def handler():
    """LangChain callback: every graph node, retriever and model call becomes an observation."""
    if not enabled():
        return None
    client()
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def trace_id_for(seed: str) -> str | None:
    if not enabled():
        return None
    from langfuse import Langfuse

    return Langfuse.create_trace_id(seed=seed)


class _Untraced:
    trace_id = None

    def update(self, **_: Any) -> None:
        pass


@contextmanager
def root(
    name: str,
    *,
    input: Any,
    trace_name: str | None = None,
    trace_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> Iterator[Any]:
    """The agent's root observation. Inside a running trace (an experiment) it nests instead."""
    if not enabled():
        yield _Untraced()
        return
    from langfuse import propagate_attributes
    from opentelemetry import trace as otel

    lf = client()
    nested = trace_id is None and otel.get_current_span().get_span_context().is_valid
    with lf.start_as_current_observation(
        as_type="agent",
        name=name,
        input=input,
        metadata=metadata,
        version=__version__,
        trace_context={"trace_id": trace_id} if trace_id else None,
    ) as observation:
        attributes = {"tags": tags, "metadata": metadata}
        if not nested:
            attributes |= {"trace_name": trace_name or name, "session_id": session_id}
        with propagate_attributes(**attributes):
            yield observation


def run_config(*, thread_id: str, callbacks: list[Any]) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks,
        "run_name": "run-graph",
    }


def score_trace(trace_id: str | None, name: str, value: str, comment: str = "") -> None:
    if not (enabled() and trace_id):
        return
    client().create_score(
        trace_id=trace_id,
        name=name,
        value=value,
        data_type="CATEGORICAL",
        comment=comment or None,
    )


def trace_url(trace_id: str | None) -> str | None:
    if not (enabled() and trace_id):
        return None
    flush()
    return client().get_trace_url(trace_id=trace_id)


def flush() -> None:
    if enabled():
        client().flush()
