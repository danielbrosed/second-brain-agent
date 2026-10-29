"""A small facade over the graph for the CLI and the experiments, and the root of every trace."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from . import tracing
from .config import Settings
from .graph import build_graph
from .llm import chat_model
from .retrieval import Chunk, MapFirstRetriever, SearchIndex, VectorIndex, VectorRetriever
from .vault import Vault


@dataclass
class AskResult:
    answer: str
    found: bool
    citations: list[str]
    warnings: list[str]
    read: list[str]
    attempts: int
    trace_id: str | None = None


@dataclass
class CaptureStep:
    thread_id: str
    pending: dict | None = None  # the draft waiting for a person, when the graph is paused
    written: list[str] = field(default_factory=list)
    error: str = ""
    trace_id: str | None = None

    def summary(self) -> dict:
        if self.error:
            return {"status": "failed", "error": self.error}
        if self.pending:
            return {"status": "awaiting review", "draft": self.pending}
        return {"status": "written" if self.written else "nothing written", "written": self.written}


class _LazyIndex:
    """Builds the embedding index on first use, so map-first runs never load the model."""

    def __init__(self, vault: Vault) -> None:
        self._vault = vault
        self._index: VectorIndex | None = None

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        if self._index is None:
            self._index = VectorIndex(self._vault)
        return self._index.search(query, k)


class Agent:
    def __init__(
        self,
        settings: Settings,
        *,
        llm: BaseChatModel | None = None,
        index: SearchIndex | None = None,
        checkpointer: Any = None,
    ) -> None:
        self.settings = settings
        self.vault = Vault(settings.vault_dir)
        llm = llm or chat_model(settings)
        method = settings.structured_method
        retrievers = {
            "map": MapFirstRetriever(
                vault=self.vault, llm=llm, structured_method=method, name="map-first-retriever"
            ),
            "vector": VectorRetriever(
                index=index or _LazyIndex(self.vault), name="vector-retriever"
            ),
        }
        self.graph = build_graph(
            self.vault,
            llm,
            retrievers=retrievers,
            structured_method=method,
            stale_after_days=settings.stale_after_days,
            checkpointer=checkpointer or InMemorySaver(),
        )

    def _invoke(self, payload: Any, thread_id: str, callbacks: list[Any]) -> dict:
        handler = tracing.handler()
        all_callbacks = [*callbacks, *([handler] if handler else [])]
        return self.graph.invoke(
            payload, tracing.run_config(thread_id=thread_id, callbacks=all_callbacks)
        )

    def ask(
        self,
        question: str,
        *,
        retriever: str = "map",
        as_of: date | None = None,
        callbacks: list[Any] | None = None,
        tags: list[str] | None = None,
    ) -> AskResult:
        as_of_text = (as_of or date.today()).isoformat()
        with tracing.root(
            "answer-question",
            input={"question": question},
            tags=["ask", f"retriever:{retriever}", *(tags or [])],
            metadata={"retriever": retriever, "as_of": as_of_text},
        ) as observation:
            state = self._invoke(
                {
                    "mode": "ask",
                    "retriever": retriever,
                    "question": question,
                    "as_of": as_of_text,
                    # Reset per-question fields in case a thread is ever reused.
                    "candidates": [],
                    "context": [],
                    "citations": [],
                    "warnings": [],
                    "attempts": 0,
                    "feedback": "",
                },
                f"ask-{uuid4().hex[:12]}",
                callbacks or [],
            )
            result = AskResult(
                answer=state.get("answer", ""),
                found=bool(state.get("found")),
                citations=state.get("citations", []),
                warnings=state.get("warnings", []),
                read=state.get("candidates", []),
                attempts=state.get("attempts", 0),
                trace_id=observation.trace_id,
            )
            observation.update(
                output={
                    "answer": result.answer,
                    "sources": result.citations,
                    "warnings": result.warnings,
                },
                metadata={
                    "found": str(result.found),
                    "notes_read": ", ".join(result.read),
                    "answer_attempts": str(result.attempts),
                },
            )
        return result

    def capture(self, request: str, *, as_of: date | None = None) -> CaptureStep:
        thread_id = f"capture-{uuid4().hex[:12]}"
        with tracing.root(
            "capture-decision",
            input={"request": request},
            trace_id=tracing.trace_id_for(thread_id),
            session_id=thread_id,
            tags=["capture"],
        ) as observation:
            state = self._invoke(
                {
                    "mode": "capture",
                    "request": request,
                    "as_of": (as_of or date.today()).isoformat(),
                    "revisions": 0,
                    "revision": "",
                    "written": [],
                },
                thread_id,
                [],
            )
            step = self._step(thread_id, state, observation.trace_id)
            observation.update(output=step.summary())
        return step

    def resume(self, thread_id: str, decision: dict) -> CaptureStep:
        """Continue a paused capture: {'action': 'approve' | 'reject' | 'revise', 'note': ...}."""
        with tracing.root(
            "review-decision",
            input=decision,
            trace_name="capture-decision",
            trace_id=tracing.trace_id_for(thread_id),
            session_id=thread_id,
            tags=["capture"],
        ) as observation:
            state = self._invoke(Command(resume=decision), thread_id, [])
            step = self._step(thread_id, state, observation.trace_id)
            observation.update(output=step.summary())
        if decision.get("action") in {"approve", "reject"}:
            # The person's call is the best label a draft can get: keep it on the trace.
            tracing.score_trace(step.trace_id, "human_review", str(decision["action"]))
        return step

    @staticmethod
    def _step(thread_id: str, state: dict, trace_id: str | None) -> CaptureStep:
        interrupts = state.get("__interrupt__") or []
        return CaptureStep(
            thread_id=thread_id,
            pending=interrupts[0].value if interrupts else None,
            written=state.get("written", []),
            error=state.get("error", ""),
            trace_id=trace_id,
        )
