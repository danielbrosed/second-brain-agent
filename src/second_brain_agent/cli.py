"""Command line for the second brain agent."""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import date

from . import tracing
from .config import Settings
from .vault import Vault, lint


def _agent(settings: Settings, persistent: bool = False):
    from .agent import Agent

    if not persistent:
        return Agent(settings), None
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
    return Agent(settings, checkpointer=SqliteSaver(conn)), conn


def cmd_ask(args, settings: Settings) -> int:
    agent, _ = _agent(settings)
    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    result = agent.ask(args.question, retriever=args.retriever, as_of=as_of)
    print(textwrap.fill(result.answer, 100) if "\n" not in result.answer else result.answer)
    if result.citations:
        print("\nSources:")
        for path in result.citations:
            print(f"  - {path}")
    if result.warnings:
        print("\nBefore you rely on it:")
        for warning in result.warnings:
            print(f"  ! {warning}")
    if url := tracing.trace_url(result.trace_id):
        print(f"\nTrace: {url}")
    return 0 if result.found else 2


def _show_pending(pending: dict) -> None:
    print(f"\nDraft for {pending['path']}")
    if pending.get("supersedes"):
        print(f"It supersedes {pending['supersedes']}")
    print("-" * 72)
    print(pending["markdown"].rstrip())
    print("-" * 72)


def _drive(agent, step, interactive: bool) -> int:
    while step.pending:
        _show_pending(step.pending)
        if not interactive:
            print(f"\nPaused. Resume with: sba resume {step.thread_id} approve|reject|revise")
            return 0
        choice = input("[a]pprove, [r]evise, or reject with any other key: ").strip().lower()
        if choice.startswith("a"):
            decision = {"action": "approve"}
        elif choice.startswith("r"):
            decision = {"action": "revise", "note": input("What should change? ").strip()}
        else:
            decision = {"action": "reject"}
        step = agent.resume(step.thread_id, decision)
    if step.error:
        print(step.error)
        return 1
    if step.written:
        print("Written: " + ", ".join(step.written))
    else:
        print("Nothing was written.")
    if url := tracing.trace_url(step.trace_id):
        print(f"Trace: {url}")
    return 0


def cmd_capture(args, settings: Settings) -> int:
    agent, conn = _agent(settings, persistent=True)
    try:
        as_of = date.fromisoformat(args.as_of) if args.as_of else None
        step = agent.capture(args.request, as_of=as_of)
        return _drive(agent, step, interactive=sys.stdin.isatty() and not args.no_input)
    finally:
        conn.close()


def cmd_resume(args, settings: Settings) -> int:
    agent, conn = _agent(settings, persistent=True)
    try:
        decision = {"action": args.action, "note": " ".join(args.note)}
        return _drive(agent, agent.resume(args.thread_id, decision), interactive=False)
    finally:
        conn.close()


def cmd_lint(args, settings: Settings) -> int:
    problems = lint(Vault(settings.vault_dir))
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problem(s)" if problems else "Vault is healthy.")
    return 1 if problems else 0


def cmd_eval(args, settings: Settings) -> int:
    from .agent import Agent
    from .evaluation import run_experiment
    from .llm import chat_model

    agent = Agent(settings)
    judge = chat_model(settings, settings.judge_model)
    retrievers = ["map", "vector"] if args.retriever == "both" else [args.retriever]
    reports = [
        run_experiment(
            agent,
            judge,
            retriever,
            model=settings.llm_model,
            limit=args.limit,
            concurrency=args.concurrency,
        )
        for retriever in retrievers
    ]
    metrics = sorted({k for r in reports for k in r["summary"] if k != "correct_by_kind"})
    kinds = sorted({k for r in reports for k in r["summary"]["correct_by_kind"]})
    print(f"\n{'metric':<28}" + "".join(f"{r['retriever']:>12}" for r in reports))
    for name in metrics:
        print(f"{name:<28}" + "".join(f"{r['summary'].get(name, '-'):>12}" for r in reports))
    for kind in kinds:
        label = f"correct · {kind}"
        values = [r["summary"]["correct_by_kind"].get(kind, "-") for r in reports]
        print(f"{label:<28}" + "".join(f"{v:>12}" for v in values))
    for report in reports:
        print(f"\n{report['run_name']}: evals/results/{report['run_name']}.json")
        if report["langfuse_url"]:
            print(f"  Langfuse: {report['langfuse_url']}")
    return 0


def cmd_prompts(args, settings: Settings) -> int:
    if not tracing.enabled():
        print("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY first.")
        return 1
    from .prompts import PROMPTS

    client = tracing.client()
    for name, messages in PROMPTS.items():
        try:
            current = client.get_prompt(name, type="chat", label="production", cache_ttl_seconds=0)
            unchanged = [(m["role"], m["content"]) for m in current.prompt] == [
                (m["role"], m["content"]) for m in messages
            ]
        except Exception:
            unchanged = False
        if unchanged:
            print(f"{name}: unchanged")
            continue
        created = client.create_prompt(
            name=name,
            prompt=messages,
            type="chat",
            labels=["production"],
            commit_message="Sync from the repository",
        )
        print(f"{name}: version {created.version} labelled production")
    return 0


def cmd_graph(args, settings: Settings) -> int:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from .graph import build_graph

    graph = build_graph(Vault(settings.vault_dir), FakeListChatModel(responses=[""]), retrievers={})
    print(graph.get_graph().draw_mermaid())
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="sba", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer a question from the vault, with sources")
    ask.add_argument("question")
    ask.add_argument("--retriever", choices=["map", "vector"], default="map")
    ask.add_argument("--as-of", help="evaluate staleness as of this date (YYYY-MM-DD)")
    ask.set_defaults(func=cmd_ask)

    capture = sub.add_parser("capture", help="draft a decision note and wait for approval")
    capture.add_argument("request")
    capture.add_argument("--as-of")
    capture.add_argument("--no-input", action="store_true", help="pause instead of prompting")
    capture.set_defaults(func=cmd_capture)

    resume = sub.add_parser("resume", help="continue a paused capture")
    resume.add_argument("thread_id")
    resume.add_argument("action", choices=["approve", "reject", "revise"])
    resume.add_argument("note", nargs="*", help="what to change, for revise")
    resume.set_defaults(func=cmd_resume)

    evaluate = sub.add_parser("eval", help="run the question set and score both retrievers")
    evaluate.add_argument("--retriever", choices=["map", "vector", "both"], default="both")
    evaluate.add_argument("--limit", type=int, help="only the first N questions")
    evaluate.add_argument("--concurrency", type=int, default=2)
    evaluate.set_defaults(func=cmd_eval)

    prompts_cmd = sub.add_parser("prompts", help="manage prompts in Langfuse")
    prompts_cmd.add_argument("action", choices=["push"])
    prompts_cmd.set_defaults(func=cmd_prompts)

    sub.add_parser("lint", help="check the vault against the method").set_defaults(func=cmd_lint)
    sub.add_parser("graph", help="print the agent graph as Mermaid").set_defaults(func=cmd_graph)

    args = parser.parse_args(argv)
    try:
        return args.func(args, Settings.from_env())
    finally:
        tracing.flush()


if __name__ == "__main__":
    raise SystemExit(main())
