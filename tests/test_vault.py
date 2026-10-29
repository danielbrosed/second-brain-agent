from datetime import date

import pytest

from second_brain_agent.vault import (
    MAP_FILE,
    VaultError,
    find_secrets,
    lint,
    parse_note,
    trust_warnings,
)


def test_demo_vault_is_healthy(vault):
    assert lint(vault) == []


def test_map_lists_every_note(vault):
    assert vault.map_paths() == set(vault.note_paths()) - {MAP_FILE}


def test_businesses_are_the_folders_with_an_index(vault):
    assert vault.businesses() == ["fieldnotes", "shared", "studio", "tidepool"]


@pytest.mark.parametrize(
    "path",
    ["../outside.md", "/etc/passwd.md", "C:/Windows/x.md", "studio/../../x.md", "studio/notes.txt"],
)
def test_read_refuses_paths_outside_the_vault(vault, path):
    with pytest.raises(VaultError):
        vault.read(path)


def test_superseded_note_points_to_its_successor(vault):
    old = vault.read("tidepool/decisions/2026-05-10-three-plans.md")
    assert old.status == "superseded"
    assert old.superseded_by() == "tidepool/decisions/2026-09-14-two-plans.md"


def test_trust_warnings():
    as_of = date(2026, 10, 30)
    stale_unverified = parse_note(
        "x.md", "---\nupdated: 2026-06-02\nconfidence: unverified\nstatus: current\n---\n# X\n"
    )
    fresh_verified = parse_note(
        "y.md", "---\nupdated: 2026-10-20\nconfidence: verified\nstatus: current\n---\n# Y\n"
    )
    snapshot = parse_note(
        "z.md", "---\nupdated: 2026-10-20\nconfidence: snapshot\nstatus: current\n---\n# Z\n"
    )
    assert len(trust_warnings(stale_unverified, as_of, 90)) == 2
    assert trust_warnings(fresh_verified, as_of, 90) == []
    assert len(trust_warnings(snapshot, as_of, 90)) == 1


def test_lint_catches_each_broken_rule(vault, vault_dir):
    fake_key = "sk_live_" + "a1B2" * 6
    note = vault_dir / "studio" / "icp.md"
    note.write_text(
        note.read_text(encoding="utf-8") + f"\nSee [[acquisition]]. {fake_key}\n", "utf-8"
    )
    (vault_dir / "studio" / "orphan.md").write_text(
        "---\nupdated: 2026-10-01\nstatus: current\ntype: operations\nbusiness: tidepool\n"
        "confidence: sure\n---\n# Orphan\n\n[gone](missing.md)\n",
        encoding="utf-8",
    )
    decision = vault_dir / "fieldnotes" / "decisions" / "2026-08-18-stays-free.md"
    text = decision.read_text(encoding="utf-8")
    decision.write_text(text.replace("status: current", "status: superseded"), encoding="utf-8")

    problems = "\n".join(lint(vault))
    assert "studio/icp.md: uses a wikilink" in problems
    assert "studio/icp.md: looks like a credential" in problems
    assert "studio/orphan.md: not listed in MAP.md" in problems
    assert "studio/orphan.md: business should be 'studio'" in problems
    assert "studio/orphan.md: confidence must be one of" in problems
    assert "studio/orphan.md: broken link missing.md" in problems
    assert "2026-08-18-stays-free.md: superseded without a valid 'Superseded by' link" in problems


def test_find_secrets_ignores_where_keys_live():
    assert find_secrets('1Password "Tidepool" → "Stripe live", env var STRIPE_SECRET_KEY') == []
    assert find_secrets("ghp_" + "x" * 30)


def test_write_decision_then_supersede_keeps_the_vault_healthy(vault):
    path = "tidepool/decisions/2026-10-30-one-plan.md"
    markdown = (
        "---\nupdated: 2026-10-30\ndate: 2026-10-30\nstatus: current\ntype: decision\n"
        "business: tidepool\nconfidence: unverified\n---\n# One plan\n\nBody.\n"
    )
    vault.write_decision("tidepool", path, markdown, "whether Tidepool has one plan")
    vault.mark_superseded("tidepool/decisions/2026-09-14-two-plans.md", path, date(2026, 10, 30))

    assert path in vault.map_paths()
    old = vault.read("tidepool/decisions/2026-09-14-two-plans.md")
    assert old.status == "superseded" and old.superseded_by() == path
    assert lint(vault) == []


def test_write_decision_refuses_overwrite_and_other_folders(vault):
    with pytest.raises(VaultError):
        vault.write_decision("studio", "studio/decisions/2026-09-03-no-cold-outreach.md", "x", "x")
    with pytest.raises(VaultError):
        vault.write_decision("studio", "studio/icp.md", "x", "x")
    with pytest.raises(VaultError):
        vault.write_decision("unknown", "unknown/decisions/a.md", "x", "x")
