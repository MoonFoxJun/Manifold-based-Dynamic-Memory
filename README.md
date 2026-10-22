# Manifold-based Dynamic Memory (MDM)

*Extractive recall over a message archive.*

Given a handful of keywords, return a short, ordered, de-duplicated passage —
never generated text — with provenance pointers back to the original messages.

This is the recall half of a long-term memory system. It deliberately does
**not** synthesise answers: selection, ordering and compression are done with
fixed functions; writing is left to a language model downstream.

> **Name vs. implementation.** The *manifold-based* index is the research
> target; v0.1.0 ships the lexical baseline that makes the gap measurable. The
> repository is honest about which is which — see *What this is (and is not)*
> and *Evaluation*.

## What this is (and is not)

| it is | it is not |
|---|---|
| a working prototype of extractive recall | a new retrieval algorithm |
| a pipeline: relevance → verdict → adaptive suppression → dedup → budgeted selection | a semantic index (the index is lexical) |
| an MCP server exposing `recall` / `remember` / `stats` | an answer generator |
| an honest evaluation harness, including its own failure modes | a benchmark or a paper |

Every component is standard information-retrieval machinery (TF-IDF, MMR,
near-duplicate clustering, recency decay, a small state machine). The
contribution, such as it is, is the composition and the measurements — and the
measurements say where the composition breaks.

## Status

Research prototype. The pipeline, the state machine and the evaluation harness
are in place; the index is lexical (TF-IDF over character bigrams), not semantic.

**No data is included in this repository.** The archive is expected at
`data/messages.jsonl` (git-ignored); bring your own with `scripts/import_chat.py`.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate  on Linux/macOS
pip install -r requirements.txt
```

## Quickstart

1. Import an archive (see `scripts/import_chat.py` for the expected format):

```bash
python scripts/import_chat.py session.jsonl data/messages.jsonl
```

2. Recall from the command line:

```bash
python -m semantic_memory.recall "leave one out hat matrix" --budget 2000
```

3. Interactive loop (`q` / `refine` / `dig` / `show` / `reset` / `finish`):

```bash
python -m semantic_memory.session
```

4. Evaluation (needs an archive):

```bash
python -m eval.eval_set
```

## MCP server (use it from an AI client)

The archive can be mounted in any MCP-capable client (Claude Desktop, Cursor,
...). The server speaks JSON-RPC 2.0 over stdio and needs no SDK:

```bash
python run_mcp.py --data data/messages.jsonl
```

Tools exposed:

| tool | kind | what it does |
|---|---|---|
| `recall(query, budget, recent, days_ago)` | read | returns an extractive passage with provenance ids; refuses honestly |
| `remember(text, role)` | **write** | appends a message to the archive and re-indexes |
| `stats()` | read | archive size and date range |

Client config (Claude Desktop `claude_desktop_config.json`); replace the paths:

```json
{
  "mcpServers": {
    "semantic-memory": {
      "command": "<path-to-python>/python.exe",
      "args": ["<path-to-repo>/run_mcp.py",
               "--data", "<path-to-repo>/data/messages.jsonl"]
    }
  }
}
```

Cursor (`.cursor/mcp.json`) uses the same shape. `run_mcp.bat` works too if you
prefer to point the client at a single executable (it uses the `python` on
`PATH`).

## How it works

```
query (keywords)
  │  TF-IDF cosine, sentence-level max for long inputs
  ▼
relevance ──► verdict: solid / weak / refuse (honest "no relevant memory")
  │  recency / freshness / access-strength weights
  ▼
crowding-adaptive suppression  (many hits per budget → compress harder)
  │
  ▼
content dedup (one representative per cluster, the rest folded into pointers)
  │
  ▼
MMR selection under a character budget → ordered passage with provenance
```

Key properties:

- **honest refusal** — a query whose best relevance is below a floor, or whose
  terms never occur in the archive, returns "no relevant memory" instead of
  noise;
- **adaptive compression** — the suppression strength scales with how many
  strong hits compete for the fixed character budget;
- **pointers, not repetition** — near-duplicate messages are folded into a
  single representative plus a list of ids that can be drilled into;
- **state across turns** — `dig` slides a time window backwards and hard-excludes
  what was already returned, so successive rounds yield new content.

## Evaluation

Two fixed query sets. Gold sets are defined by keywords present in relevant
messages, so treat the numbers as regression guards rather than a quality claim.

**Literal queries** (the query contains the gold keywords):

| metric | value |
|---|---|
| mean P@5 | 0.94 |
| mean R@5 | 0.92 |

**Paraphrase queries** (the query does *not* contain the gold keywords):

| ranker | mean P@5 |
|---|---|
| TF-IDF cosine | 0.47 |
| BM25 | 0.47 |

The gap between 0.94 and 0.47 is the point: the lexical index cannot follow a
paraphrase ("how do I estimate the error?" never mentions "leave-one-out"), and
swapping the ranker does not help. Closing that gap requires a semantic
representation, not more tuning.

## Limits

- lexical only: character bigrams match inside words (a known artefact);
- paraphrase queries fail (see Evaluation);
- no passage-level index yet: long messages dilute cosine similarity;
- no generation: a summary paragraph must come from a model downstream;
- MCP binding exists, HTTP does not; the Python function signature is the
  stable interface.

## Layout

```
semantic_memory/          # package: features, recall pipeline, session, MCP server
eval/eval_set.py          # literal evaluation set
scripts/import_chat.py    # archive importer
tests/test_recall.py      # synthetic smoke tests
docs/architecture.md      # design notes
docs/results.md           # measured numbers
run_mcp.py                # MCP entry point for clients
```

## Citation

If you use this in academic work, cite the archived release:

```
Zenodo DOI: 10.5281/zenodo.XXXXXXX   (replace after the first release)
```

A `CITATION.cff` is included; its `doi` field is a placeholder until the
release is archived by Zenodo.

## License

GPL-3.0. See `LICENSE`.
