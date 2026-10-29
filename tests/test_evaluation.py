import json

from second_brain_agent import evaluation
from second_brain_agent.agent import Agent
from second_brain_agent.evaluation import Verdict, load_items
from second_brain_agent.graph import Answer
from second_brain_agent.retrieval import RouteDecision

from .conftest import ScriptedLLM

KINDS = {
    "lookup",
    "superseded",
    "trust",
    "decision-why",
    "cross-note",
    "credential",
    "unanswerable",
}


def test_dataset_points_at_real_notes(vault):
    items = load_items()
    assert len({i["id"] for i in items}) == len(items)
    mapped = vault.map_paths()
    for item in items:
        expected = item["expected_output"]
        assert item["metadata"]["kind"] in KINDS
        assert expected["answerable"] == bool(expected["notes"]), item["id"]
        for group in expected["notes"]:
            for path in group:
                assert path in mapped, f"{item['id']}: {path}"


def test_local_run_scores_every_item(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "RESULTS_DIR", tmp_path)
    llm = ScriptedLLM(
        RouteDecision=[
            RouteDecision(notes=["studio/offer-and-pricing.md"], reason="price"),
            RouteDecision(notes=["studio/offer-and-pricing.md"], reason="credit"),
        ],
        Answer=[
            Answer(found=True, answer="€900 + VAT.", citations=["studio/offer-and-pricing.md"]),
            Answer(found=True, answer="Yes.", citations=["studio/offer-and-pricing.md"]),
        ],
    )
    judge = ScriptedLLM(
        Verdict=[Verdict(correct=True, reason="ok"), Verdict(correct=False, reason="x")]
    )

    report = evaluation.run_experiment(Agent(settings, llm=llm), judge, "map", model="m", limit=2)

    assert report["items"] == 2
    assert report["summary"]["retrieval_recall"] == 1.0
    assert report["summary"]["correct"] == 0.5
    saved = json.loads((tmp_path / f"{report['run_name']}.json").read_text(encoding="utf-8"))
    assert saved["rows"][0]["scores"]["citation_recall"] == 1.0
