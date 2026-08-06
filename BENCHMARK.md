# BENCHMARK.md — CommerceBench

This is what turns `toolrouter` from a demo into an evaluation. Without
it, "I built a semantic router" is a claim. With it, it's a measured
result.

## Dataset generation (self-play, not manual labeling)

For each `Tool` in the registry, generate 10-20 natural-language queries
that a real user might type, labeled with that tool as the ground-truth
answer. Do this by prompting an LLM with the tool's name, description,
and parameters, and asking it to produce a mix of:

- **Clean queries** — obviously map to this tool ("book a table for 4
  tonight at 8" -> `book_table`)
- **Ambiguous queries** — could plausibly map to a sibling tool too
  ("order paneer" -> could be Instamart groceries to cook, or Food
  delivery ready-to-eat — deliberately test this)
- **Typo / informal queries** — "resto near me open now"
- **Adversarial / injection-flavored queries** (optional, stretch) —
  queries that embed instruction-like text, to sanity-check the router
  doesn't do anything strange with untrusted input, even though
  full injection defense is out of scope for this project

Output format — one JSON file, one row per query:

```json
{"query": "book a table for 4 tonight", "correct_tool": "book_table", "category": "clean"}
{"query": "order paneer", "correct_tool": "search_restaurants", "category": "ambiguous"}
```

Keep `category` in the dataset — it's what lets you report accuracy
*broken down* by difficulty, which is a much more credible result than
one blended accuracy number.

## Baselines — exactly four, no more

| # | Baseline | Description |
|---|---|---|
| 1 | All tools | No retrieval. LLM sees every tool schema. This is the thing you're improving on — it needs a real number too, not just an assumption that it's worse. |
| 2 | Dense retrieval only | Embedding similarity, fixed top-k (e.g. k=5) |
| 3 | Dense + BM25 hybrid | Combine normalized dense + lexical scores |
| 4 | Dense + confidence gate | Adaptive-k as specified in `ARCHITECTURE.md` |

Do not add a fifth baseline "to be thorough." Four clean numbers beat
seven muddy ones.

## Metrics — use standard IR metrics, don't invent new ones

- **Top-1 accuracy** — is the single top-ranked tool the correct one?
- **Top-3 accuracy** — is the correct tool within the top 3?
- **MRR (Mean Reciprocal Rank)** — rewards ranking the correct tool
  higher even when it's not #1
- **Recall@k** — for k in {1, 3, 5}
- **Prompt tokens** — average tokens spent on tool schemas per query,
  compared against the "all tools" baseline (this is where your
  reduction percentage comes from — report it as measured, not assumed)
- **Latency** — p50 / p95 end-to-end retrieval time
- **Ambiguous-query behavior** (the one metric nobody else reports) —
  on rows tagged `category: ambiguous`, does the confidence gate widen k
  instead of confidently returning a single wrong answer? Report this
  separately from overall accuracy.

## Evaluation script contract

```python
# bench/evaluate.py
def evaluate(dataset_path: str, router: "ToolRouter", baseline: str) -> dict:
    """
    baseline in {"all_tools", "dense", "hybrid", "confidence_gate"}
    Returns a dict of metric_name -> float, plus a per-category
    breakdown. Write results to bench_results/<baseline>.json and
    append a row to bench_results/summary.md as a markdown table row
    (this becomes your README Results table directly - don't
    hand-transcribe numbers).
    """
```

## What "done" looks like

- `bench_results/summary.md` exists with real numbers for all four
  baselines
- The README Results table is filled in from that file, not guessed
- You can explain, for at least one ambiguous-category example, exactly
  why the confidence gate widened k (or didn't) — this is the single
  most likely thing an interviewer asks you to walk through live
