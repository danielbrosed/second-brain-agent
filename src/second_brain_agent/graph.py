"""The agent graph.

Two paths share one graph and one checkpointer:

ask      retrieve_notes -> generate_answer -> check_citations ─(uncited, once)─> generate_answer
capture  draft_decision -> await_review (interrupt: approve, reject or revise) -> write_decision

`retrieve_notes` runs whichever retriever the run asks for (map-first or vector), so the rest of
the graph is identical for both and the experiments compare retrieval and nothing else.

The model only ever returns validated objects. Reading, citation checks, trust warnings and every
write to the vault are plain code, so a bad generation can be wrong but never dangerous: it cannot
read outside the map, cite a note it did not read, or write without approval.
"""

from __future__ import annotations

import posixpath
import re
from datetime import date
from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from . import prompts
from .vault import Vault, match_path, trust_warnings

MAX_ANSWER_ATTEMPTS = 2
MAX_REVISIONS = 3
NOT_FOUND = "The vault has no note that covers this."
_ODD_SPACES = re.compile("[\u00a0\u2007\u202f]")  # models like narrow no-break spaces
UNGROUNDED = "I could not ground an answer in the vault, so I am not giving one."


class Answer(BaseModel):
    found: bool = Field(description="False when CONTEXT does not contain the answer.")
    answer: str
    citations: list[str] = Field(description="Exact paths of the notes used.")


class DecisionDraft(BaseModel):
    business: str
    slug: str
    title: str
    body: str
    supersedes: str = Field(description="MAP path of the decision this replaces, or empty.")
    map_answers: str


class AgentState(TypedDict, total=False):
    mode: Literal["ask", "capture"]
    retriever: Literal["map", "vector"]
    as_of: str
    # ask
    question: str
    candidates: list[str]
    context: list[dict]
    found: bool
    answer: str
    citations: list[str]
    warnings: list[str]
    attempts: int
    feedback: str
    # capture
    request: str
    draft: dict
    revision: str
    revisions: int
    review: dict
    written: list[str]
    error: str


def render_context(context: list[dict]) -> str:
    blocks = []
    for item in context:
        meta = item["meta"]
        header = (
            f"### {item['path']}\nupdated: {meta.get('updated')} · status: {meta.get('status')}"
            f" · confidence: {meta.get('confidence')}"
        )
        blocks.append(f"{header}\n\n{item['text'].strip()}")
    return "\n\n---\n\n".join(blocks)


def slugify(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:6]
    return "-".join(words) or "decision"


def build_graph(
    vault: Vault,
    llm: BaseChatModel,
    *,
    retrievers: dict[str, BaseRetriever],
    structured_method: str = "json_schema",
    stale_after_days: int = 90,
    checkpointer=None,
):
    def call(prompt: str, schema: type[BaseModel], step: str, variables: dict, config):
        return prompts.run_structured(
            prompt, llm, schema, structured_method, step, variables, config
        )

    # --- ask -----------------------------------------------------------------------------

    def retrieve_notes(state: AgentState, config: RunnableConfig) -> dict:
        retriever = retrievers[state.get("retriever", "map")]
        documents = retriever.invoke(state["question"], config=config)
        context = [
            {"path": d.metadata["path"], "meta": d.metadata, "text": d.page_content}
            for d in documents
        ]
        return {"candidates": list(dict.fromkeys(c["path"] for c in context)), "context": context}

    def generate_answer(state: AgentState, config: RunnableConfig) -> dict:
        attempts = state.get("attempts", 0) + 1
        if not state.get("context"):
            return {"found": False, "answer": NOT_FOUND, "citations": [], "attempts": attempts}
        result = call(
            prompts.ANSWER,
            Answer,
            "write-answer",
            {
                "context": render_context(state["context"]),
                "question": state["question"],
                "feedback": state.get("feedback", ""),
            },
            config,
        )
        return {
            "found": result.found,
            "answer": _ODD_SPACES.sub(" ", result.answer).strip(),
            "citations": result.citations,
            "attempts": attempts,
        }

    def check_citations(state: AgentState) -> dict:
        read = {item["path"] for item in state.get("context", [])}
        cited = [p for raw in state.get("citations", []) if (p := match_path(raw, read))]
        cited = list(dict.fromkeys(cited))
        if state.get("found") and not cited:
            if state.get("attempts", 0) < MAX_ANSWER_ATTEMPTS:
                return {
                    "feedback": (
                        "\n\nYour previous answer cited no path from CONTEXT. Cite the exact paths "
                        "you used, or set found to false."
                    ),
                    "citations": [],
                }
            return {
                "found": False,
                "answer": UNGROUNDED,
                "citations": [],
                "warnings": [],
                "feedback": "",
            }
        if not state.get("found"):
            # No answer means no figure to trust or check: sources and warnings would mislead.
            return {"citations": [], "warnings": [], "feedback": ""}
        as_of = date.fromisoformat(state["as_of"])
        warnings = [
            w for path in cited for w in trust_warnings(vault.read(path), as_of, stale_after_days)
        ]
        return {"citations": cited, "warnings": warnings, "feedback": ""}

    # --- capture -------------------------------------------------------------------------

    def draft_decision(state: AgentState, config: RunnableConfig) -> dict:
        previous = state.get("draft")
        feedback = ""
        if state.get("revision") and previous:
            feedback = (
                f"\n\nThe founder reviewed your previous draft and asked: {state['revision']}\n"
                f"Previous draft:\n{previous['markdown']}"
            )
        result = call(
            prompts.DRAFT_DECISION,
            DecisionDraft,
            "draft-note",
            {
                "map": vault.map_text(),
                "today": state["as_of"],
                "request": state["request"],
                "businesses": ", ".join(vault.businesses()),
                "feedback": feedback,
            },
            config,
        )
        business = result.business.strip().lower()
        if business not in vault.businesses():
            return {"error": f"The draft named an unknown business: {result.business!r}."}

        supersedes = match_path(result.supersedes, vault.map_paths()) if result.supersedes else None
        if supersedes:
            old = vault.read(supersedes)
            if old.meta.get("type") != "decision" or old.status != "current":
                supersedes = None

        slug = slugify(result.slug or result.title)
        path = f"{business}/decisions/{state['as_of']}-{slug}.md"
        n = 2
        while vault.exists(path):
            path = f"{business}/decisions/{state['as_of']}-{slug}-{n}.md"
            n += 1

        title = result.title.strip().lstrip("#").strip().splitlines()[0]
        lines = [
            "---",
            f"updated: {state['as_of']}",
            f"date: {state['as_of']}",
            "status: current",
            "type: decision",
            f"business: {business}",
            "confidence: unverified",
            "---",
            f"# {title}",
            "",
            result.body.strip(),
        ]
        if supersedes:
            link = posixpath.relpath(supersedes, posixpath.dirname(path))
            lines += ["", f"**Supersedes**: [{supersedes}]({link})"]
        return {
            "draft": {
                "path": path,
                "business": business,
                "markdown": "\n".join(lines) + "\n",
                "supersedes": supersedes or "",
                "map_answers": result.map_answers.strip() or title,
            },
            "revision": "",
        }

    def await_review(state: AgentState) -> dict:
        d = state["draft"]
        decision = interrupt(
            {"path": d["path"], "markdown": d["markdown"], "supersedes": d["supersedes"]}
        )
        action = str(decision.get("action", "reject")).lower()
        if action == "revise":
            return {
                "review": {"action": "revise"},
                "revision": str(decision.get("note", "")).strip(),
                "revisions": state.get("revisions", 0) + 1,
            }
        return {"review": {"action": "approve" if action == "approve" else "reject"}}

    def write_decision(state: AgentState) -> dict:
        d = state["draft"]
        vault.write_decision(d["business"], d["path"], d["markdown"], d["map_answers"])
        written = [d["path"]]
        if d["supersedes"]:
            vault.mark_superseded(d["supersedes"], d["path"], date.fromisoformat(state["as_of"]))
            written.append(d["supersedes"])
        return {"written": written}

    # --- wiring --------------------------------------------------------------------------

    def entry(state: AgentState) -> str:
        return "draft_decision" if state.get("mode") == "capture" else "retrieve_notes"

    def after_check(state: AgentState) -> str:
        return "generate_answer" if state.get("feedback") else END

    def after_draft(state: AgentState) -> str:
        return END if state.get("error") else "await_review"

    def after_review(state: AgentState) -> str:
        action = state["review"]["action"]
        if action == "approve":
            return "write_decision"
        if action == "revise" and state.get("revisions", 0) <= MAX_REVISIONS:
            return "draft_decision"
        return END

    graph = StateGraph(AgentState)
    for node in (
        retrieve_notes,
        generate_answer,
        check_citations,
        draft_decision,
        await_review,
        write_decision,
    ):
        graph.add_node(node)

    graph.add_conditional_edges(START, entry, ["retrieve_notes", "draft_decision"])
    graph.add_edge("retrieve_notes", "generate_answer")
    graph.add_edge("generate_answer", "check_citations")
    graph.add_conditional_edges("check_citations", after_check, ["generate_answer", END])
    graph.add_conditional_edges("draft_decision", after_draft, ["await_review", END])
    graph.add_conditional_edges(
        "await_review", after_review, ["write_decision", "draft_decision", END]
    )
    graph.add_edge("write_decision", END)
    return graph.compile(checkpointer=checkpointer)
