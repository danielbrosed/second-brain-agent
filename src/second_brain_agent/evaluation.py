"""Offline evaluation: the same questions through both retrievers, scored the same way.

With Langfuse keys, the questions are synced to a Langfuse dataset and each retriever runs as an
experiment on it, so the two runs can be compared side by side, item by item, with their traces.
Without keys, the identical loop runs locally and writes the results to evals/results/.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import date, datetime
from statistics import mean
from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models import BaseChatModel
from langfuse import Evaluation
from pydantic import BaseModel

from . import prompts, tracing
from .agent import Agent
from .config import REPO_ROOT

DATASET_FILE = REPO_ROOT / "evals" / "dataset.jsonl"
RESULTS_DIR = REPO_ROOT / "evals" / "results"
DATASET_NAME = "second-brain-qa"
# Fixed so staleness warnings, and therefore scores, do not drift as the calendar moves.
EVAL_AS_OF = date(2026, 10, 30)


class Verdict(BaseModel):
    correct: bool
    reason: str


def load_items(limit: int | None = None) -> list[dict]:
    lines = DATASET_FILE.read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    return items[:limit] if limit else items


def _field(item: Any, name: str) -> Any:
    return item[name] if isinstance(item, dict) else getattr(item, name)


# --- evaluators: each receives input, output, expected_output and metadata -------------------


def _share_hit(groups: list[list[str]], paths: list[str]) -> float | None:
    """Share of required notes found; each group lists notes that are equally good."""
    if not groups:
        return None
    return sum(any(p in paths for p in group) for group in groups) / len(groups)


def retrieval_recall(*, output, expected_output, **_):
    value = _share_hit(expected_output["notes"], output["read"])
    return [] if value is None else Evaluation(name="retrieval_recall", value=value)


def citation_recall(*, output, expected_output, **_):
    value = _share_hit(expected_output["notes"], output["citations"])
    return [] if value is None else Evaluation(name="citation_recall", value=value)


def abstention(*, output, expected_output, **_):
    ok = output["found"] == expected_output["answerable"]
    return Evaluation(name="found_matches_expected", value=1.0 if ok else 0.0)


def trust_warning(*, output, metadata, **_):
    if (metadata or {}).get("kind") != "trust":
        return []
    return Evaluation(name="trust_warning", value=1.0 if output["warnings"] else 0.0)


def tokens(*, output, **_):
    return Evaluation(name="tokens", value=float(output["tokens"]))


def make_judge(llm: BaseChatModel, method: str):
    chain = prompts.structured_chain(prompts.JUDGE, llm, Verdict, method, "judge-answer")

    def correct(*, input, output, expected_output, **_):
        verdict = chain.invoke(
            {
                "question": input["question"],
                "facts": "; ".join(expected_output["facts"]) or "(none)",
                "answerable": "yes" if expected_output["answerable"] else "no",
                "answer": output["answer"],
            }
        )
        return Evaluation(
            name="correct", value=1.0 if verdict.correct else 0.0, comment=verdict.reason
        )

    return correct


def run_averages(*, item_results, **_):
    values: dict[str, list[float]] = defaultdict(list)
    for result in item_results:
        for evaluation in result.evaluations:
            values[evaluation.name].append(float(evaluation.value))
    return [Evaluation(name=f"avg_{name}", value=mean(v)) for name, v in sorted(values.items())]


# --- running -----------------------------------------------------------------------------


def make_task(agent: Agent, retriever: str):
    def task(*, item, **_):
        usage = UsageMetadataCallbackHandler()
        started = time.perf_counter()
        result = agent.ask(
            _field(item, "input")["question"],
            retriever=retriever,
            as_of=EVAL_AS_OF,
            callbacks=[usage],
            tags=["eval"],
        )
        return {
            "answer": result.answer,
            "found": result.found,
            "citations": result.citations,
            "read": result.read,
            "warnings": result.warnings,
            "attempts": result.attempts,
            "tokens": sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values()),
            "seconds": round(time.perf_counter() - started, 2),
        }

    return task


def sync_dataset(items: list[dict]) -> None:
    """Upsert the local questions into the Langfuse dataset; item ids keep it idempotent."""
    client = tracing.client()
    try:
        client.get_dataset(DATASET_NAME)
    except Exception:
        client.create_dataset(
            name=DATASET_NAME,
            description="Questions about the fictional Northbeam vault, with the notes and facts "
            "a correct answer needs.",
        )
    for item in items:
        client.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=f"sba-{item['id']}",
            input=item["input"],
            expected_output=item["expected_output"],
            metadata=item["metadata"],
        )


def run_experiment(
    agent: Agent,
    judge_llm: BaseChatModel,
    retriever: str,
    *,
    model: str,
    limit: int | None = None,
    concurrency: int = 2,
) -> dict:
    items = load_items(limit)
    evaluators = [
        retrieval_recall,
        citation_recall,
        abstention,
        trust_warning,
        tokens,
        make_judge(judge_llm, agent.settings.structured_method),
    ]
    run_name = f"{retriever}-{model.split('/')[-1]}-{datetime.now():%Y%m%d-%H%M}"
    task = make_task(agent, retriever)
    rows: list[dict] = []
    url = None

    if tracing.enabled():
        sync_dataset(items)
        wanted = {f"sba-{i['id']}" for i in items}
        data = [i for i in tracing.client().get_dataset(DATASET_NAME).items if i.id in wanted]
        result = tracing.client().run_experiment(
            name="map-first vs vector",
            run_name=run_name,
            description=f"Retriever: {retriever}. Model: {model}.",
            data=data,
            task=task,
            evaluators=evaluators,
            run_evaluators=[run_averages],
            max_concurrency=concurrency,
            metadata={"retriever": retriever, "model": model},
        )
        url = getattr(result, "experiment_url", None)
        for r in result.item_results:
            rows.append(
                {
                    "id": r.item.id.removeprefix("sba-"),
                    "kind": (r.item.metadata or {}).get("kind"),
                    "output": r.output,
                    "scores": {e.name: e.value for e in r.evaluations},
                    "trace_id": r.trace_id,
                }
            )
        tracing.flush()
    else:
        for item in items:
            output = task(item=item)
            scores: dict[str, Any] = {}
            for evaluator in evaluators:
                found = evaluator(
                    input=item["input"],
                    output=output,
                    expected_output=item["expected_output"],
                    metadata=item["metadata"],
                )
                for e in found if isinstance(found, list) else [found]:
                    scores[e.name] = e.value
            rows.append(
                {
                    "id": item["id"],
                    "kind": item["metadata"]["kind"],
                    "output": output,
                    "scores": scores,
                }
            )

    summary = summarize(rows)
    report = {
        "run_name": run_name,
        "retriever": retriever,
        "model": model,
        "as_of": EVAL_AS_OF.isoformat(),
        "items": len(rows),
        "summary": summary,
        "langfuse_url": url,
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{run_name}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def summarize(rows: list[dict]) -> dict:
    metrics: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, value in row["scores"].items():
            metrics[name].append(float(value))
        if "correct" in row["scores"]:
            by_kind[row["kind"]].append(float(row["scores"]["correct"]))
    summary = {name: round(mean(values), 3) for name, values in sorted(metrics.items())}
    summary["seconds"] = round(mean(r["output"]["seconds"] for r in rows), 2) if rows else 0.0
    summary["correct_by_kind"] = {k: round(mean(v), 3) for k, v in sorted(by_kind.items())}
    return summary
