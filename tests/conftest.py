from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

from second_brain_agent.config import REPO_ROOT, Settings
from second_brain_agent.vault import Vault


class ScriptedLLM:
    """Stands in for a chat model: each structured call pops the next queued object."""

    def __init__(self, **queues: list) -> None:
        self.queues = {name: list(items) for name, items in queues.items()}
        self.calls: list[tuple[str, str]] = []

    def with_structured_output(self, schema, method=None):
        def run(prompt_value):
            self.calls.append((schema.__name__, prompt_value.to_string()))
            return self.queues[schema.__name__].pop(0)

        return RunnableLambda(run)

    def count(self, schema_name: str) -> int:
        return sum(1 for name, _ in self.calls if name == schema_name)


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)


@pytest.fixture
def vault_dir(tmp_path: Path) -> Path:
    target = tmp_path / "vault"
    shutil.copytree(REPO_ROOT / "vault", target)
    return target


@pytest.fixture
def vault(vault_dir: Path) -> Vault:
    return Vault(vault_dir)


@pytest.fixture
def settings(vault_dir: Path, tmp_path: Path) -> Settings:
    return Settings(
        vault_dir=vault_dir,
        llm_base_url="http://unused",
        llm_api_key="unused",
        llm_model="scripted",
        judge_model="scripted",
        structured_method="json_schema",
        stale_after_days=90,
        checkpoint_db=tmp_path / "checkpoints.db",
        langfuse_enabled=False,
    )
