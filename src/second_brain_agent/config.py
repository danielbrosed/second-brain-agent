"""Runtime settings, read once from the environment (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    vault_dir: Path
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    judge_model: str
    structured_method: str
    stale_after_days: int
    checkpoint_db: Path
    langfuse_enabled: bool

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(REPO_ROOT / ".env")
        model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        return cls(
            vault_dir=Path(os.getenv("VAULT_DIR", REPO_ROOT / "vault")),
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=model,
            judge_model=os.getenv("JUDGE_MODEL", model),
            # "json_schema" for providers with structured outputs, "function_calling" otherwise.
            structured_method=os.getenv("LLM_STRUCTURED_METHOD", "json_schema"),
            stale_after_days=int(os.getenv("STALE_AFTER_DAYS", "90")),
            checkpoint_db=Path(os.getenv("CHECKPOINT_DB", REPO_ROOT / ".cache" / "checkpoints.db")),
            langfuse_enabled=bool(
                os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
            ),
        )
