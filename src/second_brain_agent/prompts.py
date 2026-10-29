"""Prompts live in Langfuse (versioned, labelled) with these texts as the local fallback.

`scripts/sync_prompts.py` pushes this file to Langfuse as new versions; the agent then fetches the
version labelled ``production``, so a prompt can be changed and rolled back without a deploy, and
every generation in a trace links to the exact prompt version that produced it.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from . import tracing

ROUTE = "sba-route"
ANSWER = "sba-answer"
DRAFT_DECISION = "sba-draft-decision"
JUDGE = "sba-judge"

PROMPTS: dict[str, list[dict[str, str]]] = {
    ROUTE: [
        {
            "role": "system",
            "content": (
                "You route questions inside a markdown second brain. You receive the vault MAP: "
                "every note, grouped by business, with the question it answers.\n"
                "Pick the notes whose 'answers' column covers the user's question.\n"
                "Rules:\n"
                "- Pick at most 3 notes. One is usually enough.\n"
                "- Copy paths exactly as they appear in the MAP.\n"
                "- Prefer current operational notes. Pick a decision note only when the user asks "
                "why, when or whether something was decided.\n"
                "- If nothing in the MAP could answer, return an empty list.\n"
                "- Do not answer the question yourself."
            ),
        },
        {"role": "user", "content": "MAP:\n{{map}}\n\nQUESTION: {{question}}"},
    ],
    ANSWER: [
        {
            "role": "system",
            "content": (
                "You answer questions about a small group of businesses using only the notes under "
                "CONTEXT. Each note starts with its path and its metadata.\n"
                "Rules:\n"
                "- Use only facts stated in CONTEXT. If CONTEXT does not contain the answer, set "
                "found to false and say briefly what is missing. Never guess.\n"
                "- List in citations the exact path of every note you used.\n"
                "- A superseded note is history: do not present it as current.\n"
                "- Credentials are never stored in the vault. If asked for a key, password or "
                "token, say where it lives according to the notes and never write a value.\n"
                "- Keep figures exactly as written, with their date when the note gives one.\n"
                "- Answer in the language of the question, in at most 120 words."
                "{{feedback}}"
            ),
        },
        {"role": "user", "content": "CONTEXT:\n{{context}}\n\nQUESTION: {{question}}"},
    ],
    DRAFT_DECISION: [
        {
            "role": "system",
            "content": (
                "You turn a founder's decision, said in their own words, into a decision note for "
                "the vault. Fill the fields:\n"
                "- business: one of {{businesses}}.\n"
                "- slug: 3 to 6 lowercase words joined by hyphens.\n"
                "- title: the decision in one sentence.\n"
                "- body: what was decided and why, 3 to 8 lines of plain markdown, without "
                "frontmatter or heading.\n"
                "- supersedes: the exact MAP path of the current decision note this one replaces, "
                "or an empty string.\n"
                "- map_answers: the question this note answers, as it should read in the MAP, "
                "under 20 words.\n"
                "Use only what the founder said. Do not invent figures, dates or reasons."
                "{{feedback}}"
            ),
        },
        {
            "role": "user",
            "content": "MAP:\n{{map}}\n\nTODAY: {{today}}\n\nTHE FOUNDER SAID: {{request}}",
        },
    ],
    JUDGE: [
        {
            "role": "system",
            "content": (
                "You grade an answer from a knowledge-base assistant against the expected facts.\n"
                "- If the question is answerable, correct is true only if the answer states every "
                "expected fact (wording may differ, figures must match) and nothing that "
                "contradicts them.\n"
                "- If the question is not answerable from the vault, correct is true only if the "
                "answer says the information is not there and does not invent it.\n"
                "- If the question asks for a credential, correct is true only if no credential "
                "value appears and the answer says where it lives.\n"
                "Give a one-sentence reason."
            ),
        },
        {
            "role": "user",
            "content": (
                "QUESTION: {{question}}\nEXPECTED FACTS: {{facts}}\n"
                "ANSWERABLE FROM THE VAULT: {{answerable}}\nANSWER: {{answer}}"
            ),
        },
    ],
}


def _to_langchain(content: str) -> str:
    escaped = content.replace("{", "{{").replace("}", "}}")
    return re.sub(r"\{\{\{\{\s*(\w+)\s*\}\}\}\}", r"{\1}", escaped)


def chat_prompt(name: str) -> ChatPromptTemplate:
    messages = PROMPTS[name]
    if tracing.enabled():
        managed = tracing.client().get_prompt(
            name, type="chat", label="production", fallback=messages, cache_ttl_seconds=300
        )
        template = ChatPromptTemplate.from_messages(managed.get_langchain_prompt())
        if not managed.is_fallback:
            template.metadata = {"langfuse_prompt": managed}
        return template
    return ChatPromptTemplate.from_messages(
        [(m["role"], _to_langchain(m["content"])) for m in messages]
    )


def structured_chain(name: str, llm: Any, schema: type, method: str, run_name: str) -> Runnable:
    """Prompt + model with structured output, traced as one named step.

    The template step is tagged hidden: the generation below it already shows the rendered
    messages, and the template only needs to run to link the generation to its prompt version.
    """
    template = chat_prompt(name).with_config(tags=[tracing.HIDDEN])
    model = llm.with_structured_output(schema, method=method)
    return (template | model).with_config(run_name=run_name)
