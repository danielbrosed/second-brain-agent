"""The two retrieval strategies, as LangChain retrievers: a question in, documents out.

* MapFirstRetriever: the model reads MAP.md and picks up to three notes, which are returned whole.
* VectorRetriever: the classic baseline, chunk-and-embed with cosine similarity, over exactly the
  notes the router can reach (every note except MAP.md itself).

Being real retrievers, both are traced as `retriever` observations with their documents as output,
and the graph and the experiments can swap one for the other.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, ConfigDict, Field

from . import prompts
from .vault import MAP_FILE, Note, Vault, match_path

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

Embedder = Callable[[Sequence[str]], np.ndarray]


def note_document(path: str, meta: dict, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "path": path,
            "updated": str(meta.get("updated")),
            "status": meta.get("status"),
            "confidence": meta.get("confidence"),
            "type": meta.get("type"),
        },
    )


# --- map-first -----------------------------------------------------------------------------


class RouteDecision(BaseModel):
    notes: list[str] = Field(description="Note paths copied exactly from the MAP; at most 3.")
    reason: str = Field(description="One short sentence on why these notes.")


class MapFirstRetriever(BaseRetriever):
    """The model picks notes from the map; code reads them, so only mapped notes can come back."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vault: Vault
    llm: Any
    structured_method: str = "json_schema"
    max_notes: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        decision = prompts.run_structured(
            prompts.ROUTE,
            self.llm,
            RouteDecision,
            self.structured_method,
            "select-notes",
            {"map": self.vault.map_text(), "question": query},
            config={"callbacks": run_manager.get_child()},
        )
        allowed = self.vault.map_paths()
        picked = [p for raw in decision.notes if (p := match_path(raw, allowed))]

        documents: list[Document] = []
        seen: set[str] = set()

        def add(note: Note) -> None:
            if note.path not in seen:
                seen.add(note.path)
                documents.append(note_document(note.path, note.meta, note.body))

        for path in list(dict.fromkeys(picked))[: self.max_notes]:
            note = self.vault.read(path)
            add(note)
            # A superseded note never answers alone: bring its successor along.
            successor = note.superseded_by()
            if note.status == "superseded" and successor and self.vault.exists(successor):
                add(self.vault.read(successor))
        return documents


# --- vector baseline -----------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    path: str
    meta: dict
    text: str


class SearchIndex(Protocol):
    def search(self, query: str, k: int = 4) -> list[Chunk]: ...


class VectorRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Any  # a SearchIndex
    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [note_document(c.path, c.meta, c.text) for c in self.index.search(query, self.k)]


def chunk_vault(vault: Vault) -> list[Chunk]:
    """One chunk per '## ' section, each prefixed with the note title so it stands on its own."""
    chunks: list[Chunk] = []
    for path in vault.note_paths():
        if path == MAP_FILE:
            continue
        note = vault.read(path)
        sections = re.split(r"(?m)^(?=## )", note.body.strip())
        head = sections[0].strip()
        if len(sections) == 1 or len(head.splitlines()) > 1:
            first, rest = [head], sections[1:]
        else:
            first, rest = [], sections[1:]
        for section in first + rest:
            text = section.strip()
            if not text:
                continue
            if not text.startswith("# "):
                text = f"# {note.title}\n\n{text}"
            chunks.append(Chunk(path=path, meta=note.meta, text=text))
    return chunks


def fastembed_embedders(model_name: str = EMBEDDING_MODEL) -> tuple[Embedder, Embedder]:
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=model_name)

    def documents(texts: Sequence[str]) -> np.ndarray:
        return np.array(list(model.embed(list(texts))))

    def query(texts: Sequence[str]) -> np.ndarray:
        return np.array(list(model.query_embed(list(texts))))

    return documents, query


class VectorIndex:
    def __init__(
        self,
        vault: Vault,
        embed_documents: Embedder | None = None,
        embed_query: Embedder | None = None,
    ) -> None:
        if embed_documents is None or embed_query is None:
            embed_documents, embed_query = fastembed_embedders()
        self._embed_query = embed_query
        self.chunks = chunk_vault(vault)
        self._matrix = _normalize(embed_documents([c.text for c in self.chunks]))

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        q = _normalize(self._embed_query([query]))[0]
        scores = self._matrix @ q
        return [self.chunks[i] for i in np.argsort(-scores)[:k]]


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)
