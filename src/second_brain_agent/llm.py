"""Chat model factory. Any OpenAI-compatible endpoint works: Groq, OpenRouter, OpenAI, local."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .config import Settings


class ConfigError(RuntimeError):
    pass


def chat_model(settings: Settings, model: str | None = None) -> BaseChatModel:
    if not settings.llm_api_key:
        raise ConfigError("LLM_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return ChatOpenAI(
        model=model or settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=0,
        max_retries=3,
        timeout=60,
    )
