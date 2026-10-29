"""Chat model factory. Any OpenAI-compatible endpoint works: Groq, OpenRouter, OpenAI, local."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from .config import Settings


class ConfigError(RuntimeError):
    pass


def chat_model(settings: Settings, model: str | None = None) -> BaseChatModel:
    if not settings.llm_api_key:
        raise ConfigError("LLM_API_KEY is not set. Copy .env.example to .env and fill it in.")
    common = {
        "model": model or settings.llm_model,
        "api_key": settings.llm_api_key,
        "temperature": 0,
        "max_retries": 6,  # free tiers rate-limit by the minute; the client backs off on 429s
        "timeout": 60,
    }
    if "api.groq.com" in settings.llm_base_url:
        # The provider client keeps the model's reasoning (additional_kwargs.reasoning_content),
        # which the generic OpenAI client drops, so the thinking behind each step is traced.
        from langchain_groq import ChatGroq

        return ChatGroq(**common)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(base_url=settings.llm_base_url, **common)
