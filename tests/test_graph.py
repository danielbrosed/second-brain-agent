import sqlite3
from datetime import date

from langgraph.checkpoint.sqlite import SqliteSaver

from second_brain_agent.agent import Agent
from second_brain_agent.graph import NOT_FOUND, UNGROUNDED, Answer, DecisionDraft
from second_brain_agent.retrieval import Chunk, RouteDecision
from second_brain_agent.vault import lint

from .conftest import ScriptedLLM

AS_OF = date(2026, 10, 30)


def route(*notes: str) -> RouteDecision:
    return RouteDecision(notes=list(notes), reason="test")


def answer(*citations: str, found: bool = True) -> Answer:
    return Answer(found=found, answer="An answer.", citations=list(citations))


def draft(**overrides) -> DecisionDraft:
    fields = {
        "business": "tidepool",
        "slug": "gift cards in q1",
        "title": "Gift cards ship in the first quarter of 2027",
        "body": "Studios asked for them before Christmas; we ship after the Instagram link.",
        "supersedes": "",
        "map_answers": "when gift cards ship",
    }
    return DecisionDraft(**{**fields, **overrides})


# --- ask ---------------------------------------------------------------------------------


def test_map_first_reads_only_mapped_notes_and_warns(settings):
    llm = ScriptedLLM(
        RouteDecision=[route("fieldnotes/audience", "secrets/keys.md", "../outside.md")],
        Answer=[answer("fieldnotes/audience.md")],
    )
    result = Agent(settings, llm=llm).ask("How many subscribers?", as_of=AS_OF)

    assert result.read == ["fieldnotes/audience.md"]
    assert result.citations == ["fieldnotes/audience.md"]
    assert any("unverified" in w for w in result.warnings)
    assert any("150 days ago" in w for w in result.warnings)


def test_superseded_note_brings_its_successor(settings):
    llm = ScriptedLLM(
        RouteDecision=[route("studio/decisions/2026-07-20-audit-price-600.md")],
        Answer=[answer("studio/decisions/2026-10-01-audit-price-900.md")],
    )
    Agent(settings, llm=llm).ask("How much is the audit?", as_of=AS_OF)

    prompt = llm.calls[-1][1]
    assert "### studio/decisions/2026-07-20-audit-price-600.md" in prompt
    assert "### studio/decisions/2026-10-01-audit-price-900.md" in prompt


def test_answer_citing_unread_notes_is_retried_then_refused(settings):
    llm = ScriptedLLM(
        RouteDecision=[route("tidepool/support.md")],
        Answer=[answer("tidepool/made-up.md"), answer("tidepool/made-up.md")],
    )
    result = Agent(settings, llm=llm).ask("Phone support?", as_of=AS_OF)

    assert result.attempts == 2
    assert result.found is False and result.answer == UNGROUNDED
    assert "cited no path from CONTEXT" in llm.calls[-1][1]


def test_retry_that_cites_correctly_is_accepted(settings):
    llm = ScriptedLLM(
        RouteDecision=[route("tidepool/support.md")],
        Answer=[answer(), answer("tidepool/support.md")],
    )
    result = Agent(settings, llm=llm).ask("Phone support?", as_of=AS_OF)

    assert result.found and result.attempts == 2
    assert result.citations == ["tidepool/support.md"]


def test_nothing_routed_means_no_answer_call(settings):
    llm = ScriptedLLM(RouteDecision=[route()], Answer=[])
    result = Agent(settings, llm=llm).ask("What is Tidepool's churn?", as_of=AS_OF)

    assert result.answer == NOT_FOUND and not result.found
    assert llm.count("Answer") == 0


def test_not_found_answer_has_no_warnings(settings):
    llm = ScriptedLLM(
        RouteDecision=[route("tidepool/customers.md")],
        Answer=[answer(found=False)],
    )
    result = Agent(settings, llm=llm).ask("What is Tidepool's churn?", as_of=AS_OF)
    assert not result.found and result.warnings == []


class FakeIndex:
    def __init__(self, vault):
        note = vault.read("tidepool/offer-and-pricing.md")
        self.chunk = Chunk(path=note.path, meta=note.meta, text="Starter €29/month + VAT")

    def search(self, query, k=4):
        return [self.chunk]


def test_vector_retriever_answers_from_chunks(settings, vault):
    llm = ScriptedLLM(Answer=[answer("tidepool/offer-and-pricing.md")])
    agent = Agent(settings, llm=llm, index=FakeIndex(vault))
    result = agent.ask("Starter price?", retriever="vector", as_of=AS_OF)

    assert result.citations == ["tidepool/offer-and-pricing.md"]
    assert llm.count("RouteDecision") == 0
    assert "Starter €29/month + VAT" in llm.calls[-1][1]


# --- capture -----------------------------------------------------------------------------


def test_capture_pauses_and_writes_only_after_approval(settings, vault):
    agent = Agent(settings, llm=ScriptedLLM(DecisionDraft=[draft()]))
    step = agent.capture("Gift cards will ship in Q1 2027.", as_of=AS_OF)

    path = "tidepool/decisions/2026-10-30-gift-cards-in-q1.md"
    assert step.pending["path"] == path
    assert not vault.exists(path)

    step = agent.resume(step.thread_id, {"action": "approve"})
    assert step.written == [path]
    assert path in vault.map_paths()
    assert vault.read(path).confidence == "unverified"
    assert lint(vault) == []


def test_rejected_draft_writes_nothing(settings, vault):
    agent = Agent(settings, llm=ScriptedLLM(DecisionDraft=[draft()]))
    step = agent.capture("Gift cards will ship in Q1 2027.", as_of=AS_OF)
    before = vault.map_text()

    step = agent.resume(step.thread_id, {"action": "reject"})
    assert step.written == [] and step.pending is None
    assert vault.map_text() == before


def test_revision_loops_back_to_the_drafter(settings, vault):
    llm = ScriptedLLM(DecisionDraft=[draft(), draft(slug="gift cards in january")])
    agent = Agent(settings, llm=llm)
    step = agent.capture("Gift cards will ship in Q1 2027.", as_of=AS_OF)

    step = agent.resume(step.thread_id, {"action": "revise", "note": "say January"})
    assert step.pending["path"].endswith("gift-cards-in-january.md")
    assert "asked: say January" in llm.calls[-1][1]

    step = agent.resume(step.thread_id, {"action": "approve"})
    assert step.written == ["tidepool/decisions/2026-10-30-gift-cards-in-january.md"]


def test_superseding_decision_marks_the_old_one(settings, vault):
    old = "tidepool/decisions/2026-09-14-two-plans.md"
    agent = Agent(
        settings, llm=ScriptedLLM(DecisionDraft=[draft(supersedes=old.removesuffix(".md"))])
    )
    step = agent.capture("We go back to one plan.", as_of=AS_OF)
    assert step.pending["supersedes"] == old

    step = agent.resume(step.thread_id, {"action": "approve"})
    assert vault.read(old).status == "superseded"
    assert vault.read(old).superseded_by() == step.written[0]
    assert lint(vault) == []


def test_draft_cannot_supersede_an_operational_note(settings):
    agent = Agent(
        settings, llm=ScriptedLLM(DecisionDraft=[draft(supersedes="tidepool/support.md")])
    )
    step = agent.capture("Phone support starts in December.", as_of=AS_OF)
    assert step.pending["supersedes"] == ""


def test_unknown_business_ends_without_writing(settings):
    agent = Agent(settings, llm=ScriptedLLM(DecisionDraft=[draft(business="bakery")]))
    step = agent.capture("Open a bakery.", as_of=AS_OF)
    assert step.pending is None and step.written == []
    assert "unknown business" in step.error


def test_pause_survives_a_restart(settings, vault):
    def agent_with_db(llm):
        conn = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
        return Agent(settings, llm=llm, checkpointer=SqliteSaver(conn)), conn

    first, conn = agent_with_db(ScriptedLLM(DecisionDraft=[draft()]))
    step = first.capture("Gift cards will ship in Q1 2027.", as_of=AS_OF)
    conn.close()

    second, conn = agent_with_db(ScriptedLLM())
    try:
        step = second.resume(step.thread_id, {"action": "approve"})
    finally:
        conn.close()
    assert step.written == ["tidepool/decisions/2026-10-30-gift-cards-in-q1.md"]
