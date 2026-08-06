# BUILD_PLAN.md

Three weekends. Each ends with something that actually runs — not a
partial feature that only works if you squint.

## Weekend 1 — parsing + indexing + retrieval

**Goal:** given a manifest and a query, get back a ranked list of tools.
No confidence gate, no benchmark, no agent integration yet.

- [ ] `parser/manifest_parser.py` — parse `examples/swiggy_manifest.json`
      into `list[Tool]`
- [ ] `parser/tool_registry.py` — wrap it in a queryable registry
- [ ] `index/embed.py` — embedding model wrapper (with offline fallback)
- [ ] `index/vector_store.py` — FAISS or numpy brute-force fallback
- [ ] `router/retrieve.py` — dense-only retrieval, no hybrid yet
- [ ] `examples/quickstart.py` runs end to end and prints ranked tools
      for 3-4 hardcoded test queries
- [ ] `tests/test_manifest_parser.py` passes

**Done when:** you can run `python examples/quickstart.py` and see
sensible top-5 tools for an obvious query like "book a table tonight."

## Weekend 2 — confidence gate + explainability + real agent wiring

**Goal:** adaptive-k routing, human-readable explanations, and an actual
agent that only sees the routed subset (not all 35 tools).

- [ ] `router/confidence_gate.py` — implement the gap-based adaptive-k
      logic from `ARCHITECTURE.md`
- [ ] `router/explain.py` — per-candidate explanation dicts
- [ ] `router/prompt_builder.py` — minimal tool-schema injection
- [ ] Wire into an agent runtime (OpenAI Agents SDK is the cleanest MCP
      integration) so a real query -> routed tools -> actual tool call
      loop works
- [ ] Test against the real Swiggy MCP manifest if you have localhost
      access by this point; otherwise keep testing against the mock
- [ ] `tests/test_confidence_gate.py` — specifically test the two edge
      cases called out in `ARCHITECTURE.md` (fewer than 2 candidates,
      all-scores-below-floor)

**Done when:** you can show one query where the gate confidently
returns top-1, and one deliberately ambiguous query where it widens k —
and explain why, live, without looking at the code.

## Weekend 3 — CommerceBench + results + README + demo

**Goal:** the thing that makes this a benchmark, not a demo.

- [ ] `bench/generate_dataset.py` — self-play query generation per
      `BENCHMARK.md`, covering clean / ambiguous / typo categories
- [ ] `bench/baselines.py` — implement all four baselines, no more
- [ ] `bench/evaluate.py` — compute the metrics list from `BENCHMARK.md`,
      write to `bench_results/summary.md`
- [ ] Fill in the README Results table from real output
- [ ] Add BM25 hybrid (`index/bm25.py`) as baseline #3 **only if** time
      remains after the above — this is the first thing to cut if you're
      behind schedule, not the manifest parser or the gate
- [ ] Record one 60-90 second terminal recording or GIF: query in ->
      retrieved tools with scores -> agent calls the right tool
- [ ] Final README pass — Problem / Architecture / Method / Benchmark /
      Results / Limitations / Future Work, in that order, no marketing
      language

**Done when:** a stranger can clone the repo, read the README top to
bottom, and understand exactly what problem this solves and how well it
solves it — without opening a single `.py` file.

## If you fall behind

Cut in this order (first to cut is first on this list):

1. BM25 hybrid baseline
2. Real agent-runtime wiring (keep testing at the retrieval-only level)
3. Ambiguous/adversarial dataset categories (keep just "clean" queries)
4. Demo GIF (a results table is more important than a video)

Never cut: the manifest parser being genuinely generic, the confidence
gate, and the results table with real numbers. Those three are the
entire point of the project.
