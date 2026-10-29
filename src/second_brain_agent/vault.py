"""The vault: plain markdown notes with a small, strict frontmatter and one flat map.

Conventions (see vault/README.md for the full method):

* Every note carries the same frontmatter keys; ``date`` only on decisions.
* ``MAP.md`` lists every note with the question it answers, grouped by business, so an
  agent can jump to the right note in one hop instead of crawling folders.
* Links are relative markdown links, never wikilinks, so a raw reader can resolve them.
* A decision is never deleted: the old note becomes ``superseded`` and links to its successor.
* The vault holds the map of where credentials live, never a credential.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

MAP_FILE = "MAP.md"
SHARED = "shared"

STATUSES = {"current", "superseded"}
CONFIDENCES = {"verified", "snapshot", "unverified"}
TYPES = {
    "index", "decision", "operations", "offer", "icp", "acquisition", "clients",
    "platforms", "communications", "legal", "archive",
}  # fmt: skip
REQUIRED_KEYS = {"updated", "status", "type", "business", "confidence"}
OPTIONAL_KEYS = {"date", "sources"}

_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", re.DOTALL)
_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+\.md)\)")
_SECTION = re.compile(r"^##\s+.*\(`([a-z0-9-]+)`\)\s*$")
_SUPERSEDED_BY = re.compile(r"\*\*Superseded by\*\*:\s*\[[^\]]*\]\(([^)\s]+\.md)\)")
_SECRET_PATTERNS = [
    re.compile(p)
    for p in (
        r"sk_(live|test)_[A-Za-z0-9]{10,}",
        r"(rk|pk)_live_[A-Za-z0-9]{10,}",
        r"\bsk-[A-Za-z0-9_-]{20,}",
        r"\b[ps]k-lf-[A-Za-z0-9-]{10,}",
        r"\bgh[pousr]_[A-Za-z0-9]{20,}",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    )
]


class VaultError(ValueError):
    """A path or write the vault refuses."""


@dataclass(frozen=True)
class Note:
    path: str  # posix path relative to the vault root
    meta: dict
    body: str

    @property
    def title(self) -> str:
        for line in self.body.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return self.path

    @property
    def status(self) -> str:
        return str(self.meta.get("status", ""))

    @property
    def confidence(self) -> str:
        return str(self.meta.get("confidence", ""))

    @property
    def updated(self) -> date | None:
        value = self.meta.get("updated")
        return value if isinstance(value, date) else None

    def superseded_by(self) -> str | None:
        match = _SUPERSEDED_BY.search(self.body)
        return resolve_link(self.path, match.group(1)) if match else None


@dataclass(frozen=True)
class MapEntry:
    path: str
    answers: str
    business: str


def parse_note(path: str, text: str) -> Note:
    match = _FRONTMATTER.match(text)
    if not match:
        return Note(path=path, meta={}, body=text)
    meta = yaml.safe_load(match.group(1)) or {}
    return Note(path=path, meta=meta if isinstance(meta, dict) else {}, body=match.group(2))


def resolve_link(from_path: str, link: str) -> str | None:
    """Resolve a relative link found in ``from_path``; None if it escapes the vault."""
    if link.startswith(("/", "\\")) or "://" in link or ":" in link.split("/")[0]:
        return None
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(from_path), link))
    return None if joined.startswith("..") else joined


def match_path(raw: str, allowed: set[str]) -> str | None:
    """Map a path written by a model (quoted, without .md, with backslashes) onto an allowed one."""
    path = raw.strip().strip("`'\"").removeprefix("./").replace("\\", "/")
    for candidate in (path, f"{path}.md"):
        if candidate in allowed:
            return candidate
    return None


def business_of(path: str) -> str:
    head, _, rest = path.partition("/")
    return head if rest else SHARED


class Vault:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        if not (self.root / MAP_FILE).is_file():
            raise VaultError(f"{self.root} has no {MAP_FILE}")

    # --- reading -----------------------------------------------------------------------

    def _file(self, path: str) -> Path:
        clean = posixpath.normpath(path.replace("\\", "/"))
        if clean.startswith(("..", "/")) or not clean.endswith(".md") or ":" in clean:
            raise VaultError(f"refusing path outside the vault: {path!r}")
        target = (self.root / clean).resolve()
        if not target.is_relative_to(self.root):
            raise VaultError(f"refusing path outside the vault: {path!r}")
        return target

    def exists(self, path: str) -> bool:
        try:
            return self._file(path).is_file()
        except VaultError:
            return False

    def read(self, path: str) -> Note:
        target = self._file(path)
        if not target.is_file():
            raise VaultError(f"no such note: {path!r}")
        rel = target.relative_to(self.root).as_posix()
        return parse_note(rel, target.read_text(encoding="utf-8"))

    def map_text(self) -> str:
        return (self.root / MAP_FILE).read_text(encoding="utf-8")

    def map_entries(self) -> list[MapEntry]:
        entries: list[MapEntry] = []
        section = SHARED
        for line in parse_note(MAP_FILE, self.map_text()).body.splitlines():
            if heading := _SECTION.match(line):
                section = heading.group(1)
            elif line.startswith("|") and (link := _LINK.search(line)):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                answers = cells[1] if len(cells) > 1 else ""
                entries.append(MapEntry(path=link.group(2), answers=answers, business=section))
        return entries

    def map_paths(self) -> set[str]:
        return {e.path for e in self.map_entries()}

    def businesses(self) -> list[str]:
        return sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and (p / "INDEX.md").is_file()
        )

    def note_paths(self) -> list[str]:
        """Every note an agent may read: all markdown except templates and the vault README."""
        paths = []
        for file in sorted(self.root.rglob("*.md")):
            rel = file.relative_to(self.root).as_posix()
            if rel.startswith(("_templates/", ".")) or rel == "README.md":
                continue
            paths.append(rel)
        return paths

    # --- writing -----------------------------------------------------------------------

    def write_decision(self, business: str, path: str, markdown: str, answers: str) -> None:
        """Create a decision note and list it in the map. Never overwrites."""
        if business not in self.businesses():
            raise VaultError(f"unknown business: {business!r}")
        if not path.startswith(f"{business}/decisions/"):
            raise VaultError(f"decisions live in {business}/decisions/, not {path!r}")
        target = self._file(path)
        if target.exists():
            raise VaultError(f"refusing to overwrite {path!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown, encoding="utf-8")
        self._add_map_row(business, path, answers)

    def mark_superseded(self, old_path: str, new_path: str, as_of: date) -> None:
        target = self._file(old_path)
        text = target.read_text(encoding="utf-8")
        match = _FRONTMATTER.match(text)
        if not match:
            raise VaultError(f"{old_path!r} has no frontmatter")
        front = re.sub(r"(?m)^status:.*$", "status: superseded", match.group(1))
        front = re.sub(r"(?m)^updated:.*$", f"updated: {as_of.isoformat()}", front)
        link = posixpath.relpath(new_path, posixpath.dirname(old_path) or ".")
        body = match.group(2).rstrip("\n") + f"\n\n**Superseded by**: [{new_path}]({link})\n"
        target.write_text(f"---\n{front}\n---\n{body}", encoding="utf-8")

    def _add_map_row(self, business: str, path: str, answers: str) -> None:
        lines = self.map_text().splitlines()
        row = f"| [{path.removesuffix('.md')}]({path}) | {answers.replace('|', '/')} |"
        in_section = False
        for i, line in enumerate(lines):
            if heading := _SECTION.match(line):
                in_section = heading.group(1) == business
            elif in_section and re.match(r"^\|\s*-+", line):
                lines.insert(i + 1, row)
                (self.root / MAP_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
                return
        raise VaultError(f"MAP.md has no table for business {business!r}")


def trust_warnings(note: Note, as_of: date, stale_after_days: int) -> list[str]:
    """What a reader should know before relying on a note: the vault answers, the source decides."""
    warnings = []
    if note.confidence == "unverified":
        warnings.append(f"{note.path} is unverified: stated by the founder, no written source.")
    elif note.confidence == "snapshot":
        warnings.append(
            f"{note.path} is a snapshot copied from its source; check the source before "
            "publishing or charging this figure."
        )
    if note.updated and (age := (as_of - note.updated).days) > stale_after_days:
        warnings.append(f"{note.path} was last updated {age} days ago.")
    if note.status == "superseded":
        warnings.append(f"{note.path} is superseded and no longer current.")
    return warnings


def find_secrets(text: str) -> list[str]:
    return [m.group(0)[:12] + "…" for p in _SECRET_PATTERNS for m in p.finditer(text)]


def redact_secrets(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def lint(vault: Vault) -> list[str]:
    """Every rule of the method that a machine can check. Empty list means a healthy vault."""
    problems: list[str] = []
    paths = vault.note_paths()
    listed = vault.map_paths()

    for entry in sorted(listed - set(paths)):
        problems.append(f"MAP.md lists {entry} but the note does not exist")

    for path in paths:
        note = vault.read(path)
        meta, text = note.meta, vault._file(path).read_text(encoding="utf-8")
        where = f"{path}:"
        if not meta:
            problems.append(f"{where} missing frontmatter")
            continue
        keys = set(meta)
        if missing := REQUIRED_KEYS - keys:
            problems.append(f"{where} missing keys {sorted(missing)}")
        if extra := keys - REQUIRED_KEYS - OPTIONAL_KEYS:
            problems.append(f"{where} unexpected keys {sorted(extra)}")
        if meta.get("status") not in STATUSES:
            problems.append(f"{where} status must be one of {sorted(STATUSES)}")
        if meta.get("confidence") not in CONFIDENCES:
            problems.append(f"{where} confidence must be one of {sorted(CONFIDENCES)}")
        if meta.get("type") not in TYPES:
            problems.append(f"{where} unknown type {meta.get('type')!r}")
        if not isinstance(meta.get("updated"), date):
            problems.append(f"{where} updated must be a YYYY-MM-DD date")
        is_decision = meta.get("type") == "decision"
        if is_decision != isinstance(meta.get("date"), date):
            problems.append(f"{where} 'date' is required on decisions and only on decisions")
        if "/decisions/" in path and not is_decision:
            problems.append(f"{where} notes under decisions/ must have type: decision")
        if meta.get("business") != business_of(path):
            problems.append(f"{where} business should be {business_of(path)!r}")
        if path != MAP_FILE and path not in listed:
            problems.append(f"{where} not listed in MAP.md")
        if "[[" in text:
            problems.append(f"{where} uses a wikilink; use relative markdown links")
        for _, link in _LINK.findall(note.body):
            target = resolve_link(path, link)
            if target is None or not vault.exists(target):
                problems.append(f"{where} broken link {link}")
        if note.status == "superseded":
            successor = note.superseded_by()
            if not successor or not vault.exists(successor):
                problems.append(f"{where} superseded without a valid 'Superseded by' link")
        for secret in find_secrets(text):
            problems.append(f"{where} looks like a credential ({secret}); store where it lives")
    return problems
