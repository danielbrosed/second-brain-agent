# second-brain-agent

An agent that answers questions from a markdown second brain, cites the notes it used, tells
you how far to trust them, and records new decisions only after a person approves them.
Built with **LangGraph**, traced and evaluated with **Langfuse**.

The vault is the method I use to run my own businesses, filled here with a fictional company
so it can be public.

## What it demonstrates

**LangGraph**
- One `StateGraph` with two paths (ask and capture) and conditional routing between them.
- A bounded retry cycle: an answer that cites nothing it read gets one more attempt, then the
  agent declines instead of guessing.
- Human in the loop with `interrupt()`. A drafted decision waits for approve, reject or "revise
  with this note", which loops back to the drafter. Checkpoints live in SQLite, so a pending
  approval survives a restart (`sba capture`, close the terminal, `sba resume <thread>`).

**Langfuse**
- One trace per unit of work, following Langfuse's tracing best practices:
  - an `agent` root with a stable, verb-first name;
  - the question and the answer as the trace's input and output, not the raw graph state;
  - retrieval traced as `retriever` observations whose output is the notes it returned;
  - every model call as a generation with its tokens.
- A captured decision stays one trace across the pause for review, even when it is resumed
  from another process: its trace id is derived from the LangGraph thread id.
- Anything shaped like a credential is masked before it leaves the process: the same rule the
  vault linter enforces on the notes.
- Prompts versioned in Langfuse and fetched by label, with the repository copy as the
  fallback. Each generation links to the exact prompt version that produced it.
- A 44-question dataset and experiments that run two retrieval strategies on it, scored by
  deterministic evaluators and an LLM judge.
- The person's approve or reject on a drafted decision, stored as a score on its trace.

**Guardrails in code, not in the prompt.** The model only returns validated objects: it has no
tools and never touches the filesystem. Plain code decides which files can be read (only
notes listed in the map), drops citations of notes that were not read, computes the trust
warnings and performs every write, only after approval.

## The idea: map-first retrieval

The usual approach to "chat with your notes" is to chunk them, embed the chunks and retrieve
the nearest ones. This vault already has a better index: [`MAP.md`](vault/MAP.md) lists every
note with the question it answers. The router reads the map, picks at most three notes and
the agent reads them whole.

Reading whole notes keeps what chunks lose:

- **Trust.** Each note says when it was updated and how far to trust it: `verified`,
  `snapshot` or `unverified`. The answer carries that warning.
- **History.** A superseded decision always brings its successor along, so an old price is
  never quoted as current.
- **One hop and nothing to rebuild.** Adding a note means adding a row to the map.

The cost is the map itself, read on every question. The experiments measure whether that
trade is worth it against a standard embedding retriever (`BAAI/bge-small-en-v1.5`, top 4
chunks) over the same notes.

## How it works

```mermaid
flowchart LR
    S([question or decision]) --> E{mode}
    E -- ask --> R[retrieve_notes<br/>map-first or vector retriever]
    R --> A[generate_answer<br/>LLM, structured]
    A --> C[check_citations<br/>code: citations ⊆ notes read,<br/>trust warnings]
    C -- cited nothing, first time --> A
    C --> X([answer + sources + warnings])
    E -- capture --> D[draft_decision<br/>LLM fills the decision template]
    D --> H{{await_review<br/>interrupt: a person decides}}
    H -- revise --> D
    H -- approve --> W[write_decision<br/>code: new note, map row,<br/>mark the old one superseded]
    H -- reject --> Y([nothing written])
```

Both retrieval strategies are LangChain retrievers, so the graph is identical for both and the
experiments swap only the retriever:

- **`MapFirstRetriever`**: one generation (`select-notes`) picks up to three paths from the map.
  Code then keeps only mapped paths, reads those notes whole and follows superseded links.
- **`VectorRetriever`**: the top four chunks by cosine similarity, over the same notes.

`sba graph` prints the compiled graph as Mermaid.

## Results

### Calibrating the judge first

An LLM judge is only worth its scores if it agrees with a person. I labelled by hand the 44
answers from the first map-first run (correct if it answers what the question asks without
contradicting or inventing anything) and measured the judge, `gpt-oss-20b`, against those labels
as a Langfuse experiment on its own dataset, `judge-calibration/answer-correctness`.

| Judge prompt | Rule | Agreement with the human labels |
|---|---|---|
| v1 | the answer must state every expected fact | 65.1% (28 of 43) |
| v2 | grade what the question asks for; contradicting any fact fails | **97.7% (42 of 43)** |

Version 1 failed correct answers for leaving out details the question never asked for ("Do we
send cold emails?" → "No" failed for not adding "nor cold LinkedIn messages"). Version 2 changes
that one rule. The remaining disagreement is a label I marked as debatable. The labels and their
notes are in [`evals/judge_labels.jsonl`](evals/judge_labels.jsonl).

The same run surfaced two real agent failures:

- Asked for a credential, the agent said where it lives but flagged the answer as not found.
  Fixed in the answer prompt (v2), for both retrievers.
- "What is Tidepool's monthly recurring revenue?" found nothing, because the map row says
  "MRR". This one is left unfixed on purpose: tuning the router to a test question would only
  inflate the score. In the final run below the router found it, so the honest reading is
  that routing on the map's wording is unstable, not that it was fixed.

### Map-first vs vector

44 questions, both retrievers with the same prompt versions, agent `gpt-oss-120b`, calibrated
judge `gpt-oss-20b`, evaluated as of 2026-10-30. Each run is a Langfuse experiment on the
`second-brain-qa` dataset; the raw results are in [`evals/results/`](evals/results/).

| | Map-first | Vector (top 4 chunks) |
|---|---|---|
| Correct, by the calibrated judge | **97.7%** (42 of 43) | 95.1% (39 of 41) |
| Read the notes the answer needs | **100%** | 97.6% |
| Cited the notes the answer needs | **100%** | 96.4% |
| Answered, or declined, when it should | **100%** | 95.5% |
| Trust warning on snapshot or unverified notes | 100% | 100% |
| Tokens per question | 2,640 | **1,108** |
| Seconds per question | 11.7 | **5.7** |

- **Where vector failed, chunks were the cause.** For a question that needs two notes, the top
  four chunks missed one of them. For "Which client is on the care plan?", the chunk that
  mentions the care plan did not carry the client's name. Map-first reads notes whole, so it
  missed neither.
- **The one map-first failure is a debatable label**: listing the roadmap without saying it is
  not committed.
- **The price of map-first is the map**, read on every question: 2.4 times the tokens and twice
  the time, on Groq's free tier.
- **At 31 notes the gap in correctness is small**, two failures against one. Map-first buys
  reliability on multi-note and whole-note questions; vector is cheaper and faster. Which
  is worth it depends on how costly a wrong answer is.
- The judge returned invalid JSON on one map-first answer and three vector answers, even after
  a retry; those are left unscored rather than guessed.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env      # add LLM_API_KEY; the Langfuse keys are optional
uv run sba lint           # the vault follows its own rules
uv run sba ask "Can studios write notes about a member's injuries?"
uv run sba ask "How many subscribers does Field Notes have?"    # comes with a warning
uv run sba capture "Tidepool gift cards ship in the first quarter of 2027"
```

Any OpenAI-compatible endpoint works; the default is Groq's free tier with `openai/gpt-oss-120b`.
With Langfuse keys set:

```bash
uv run sba prompts push           # version the prompts in Langfuse, labelled production
uv run sba eval --retriever both  # dataset + two experiments, results in evals/results/
```

Tests run without network or keys, against a scripted model: `uv run pytest`.

## The vault

[`vault/`](vault/) opens in Obsidian or any editor. The rules are in its
[README](vault/README.md):

- one flat map;
- seven frontmatter keys;
- append-only decisions;
- relative links;
- where each key lives, never the key.

`sba lint` checks all of them, including a scan for anything that looks like a credential, and
it runs in CI.

## Design choices

- **The caller already knows the intent.** Ask and capture are separate commands, so routing
  between them is a plain conditional edge, not a model call.
- **Every loop is bounded.** One retry for an uncited answer, three revisions for a draft.
- **Structured outputs everywhere.** Each model call returns a Pydantic object. Free text only
  appears in the final answer.
- **A fixed evaluation date.** Staleness warnings depend on the date, so experiments evaluate
  "as of 2026-10-30" and stay comparable over time.
- **Model-agnostic.** One `ChatOpenAI` client pointed at any compatible endpoint. The judge model
  is configurable separately.

## Limitations

- The vault is small (31 notes) and fictional. At this size embeddings find the right note for
  41 of the 42 answerable questions, so the comparison is about answer quality, trust handling
  and cost more than about recall.
- The judge is a language model. Its verdicts are spot-checked by hand, not ground truth.
- The map grows with the vault. Beyond a few hundred notes it should be split by business and
  routed in two hops.

## Layout

```text
src/second_brain_agent/
  graph.py        the LangGraph graph: nodes, edges, interrupt
  agent.py        facade used by the CLI and the experiments; root of every trace
  vault.py        reading, the map, safe writes and the linter
  retrieval.py    map-first and vector retrievers
  prompts.py      prompts, synced to Langfuse prompt management
  evaluation.py   evaluators, LLM judge, Langfuse experiments
  tracing.py      Langfuse client, masking, root observations, scores
  cli.py          sba ask | capture | resume | eval | prompts | lint | graph
vault/            the fictional second brain
evals/            dataset.jsonl and the results of each run
tests/            33 tests, scripted model, no network
```

## License

[MIT](LICENSE)
