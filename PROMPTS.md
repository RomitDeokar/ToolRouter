# PROMPTS.md

Ready-to-paste prompts for an AI coding assistant (Claude Code, Cursor,
etc.), in build order. Each one references `ARCHITECTURE.md` and
`BENCHMARK.md` directly so the assistant implements against the same
contracts instead of improvising its own design.

**How to use this file:** open the repo in your AI coding tool, make
sure it can read `ARCHITECTURE.md`, `BENCHMARK.md`, and `BUILD_PLAN.md`
(point it at them explicitly if it doesn't auto-discover), then paste
one prompt at a time, in order. Review each diff before moving to the
next prompt — don't batch all of them and hope.

---

### Prompt 1 — manifest parser

```
Read ARCHITECTURE.md, section "parser/manifest_parser.py". Implement
that exact Tool dataclass and parse_manifest() function in
toolrouter/parser/manifest_parser.py.

Requirements:
- Must be generic across any MCP manifest, not hardcoded to Swiggy's
  field names specifically.
- Accept either a local file path or an http(s) URL.
- Raise a clear, specific error (not a bare KeyError) if required
  fields are missing from a tool entry.
- Add a docstring example showing both input shapes it should handle.

Also create examples/swiggy_manifest.json - a small MOCK manifest (8-10
tools) covering all three servers (food, instamart, dineout) with
realistic-looking but clearly fictional example tool names and
descriptions, so I can test locally. Label it clearly in a comment/README
note as a mock for local dev, not the real verified Swiggy manifest.

Write tests/test_manifest_parser.py covering: valid manifest parses
correctly, missing required field raises a clear error, embedding text
includes parameter names not just the description.
```

---

### Prompt 2 — tool registry

```
Read ARCHITECTURE.md, section "parser/tool_registry.py". Implement
ToolRegistry in toolrouter/parser/tool_registry.py exactly to that
contract: by_server(), by_name(), and a .tools property.

Keep it dependency-free - plain Python, no database. This is an
in-memory registry for a single-process tool.
```

---

### Prompt 3 — embeddings

```
Read ARCHITECTURE.md, section "index/embed.py". Implement EmbeddingModel
in toolrouter/index/embed.py.

Primary path: use fastembed (or sentence-transformers if fastembed isn't
available in this environment - pick one and be consistent).

Required fallback: if the embedding library or its model weights can't
be loaded (no network, no local cache), fall back to a deterministic
hash-based vector (e.g. hash each token, project into a fixed-size
vector, normalize) so the rest of the pipeline is testable offline. Log
a clear warning when the fallback is active - this must never fail
silently, since fallback-quality embeddings would quietly wreck the
benchmark numbers if someone forgot they were in fallback mode.

Expose .dim as an integer property, embed_text(text) -> np.ndarray, and
embed_batch(texts: list[str]) -> np.ndarray (2D array).
```

---

### Prompt 4 — vector store

```
Read ARCHITECTURE.md, section "index/vector_store.py". Implement
VectorStore in toolrouter/index/vector_store.py.

Try to import faiss; if unavailable, fall back to a brute-force numpy
cosine-similarity search over an in-memory array. Both code paths must
produce the same output shape: add(ids, vectors) and
search(query_vector, k) -> list[(id, score)] sorted descending by score.

Write a quick test that adds 5 known vectors and confirms search()
returns the closest one first for a query vector identical to one of
the 5.
```

---

### Prompt 5 — retrieval

```
Read ARCHITECTURE.md, section "router/retrieve.py" and the ScoredTool /
RouteResult dataclasses. Implement Retriever in
toolrouter/router/retrieve.py.

retrieve(query, k=5, hybrid=False):
- Embed the query with the same EmbeddingModel used to build the index.
- Dense-only by default.
- If hybrid=True and a BM25Index was provided, combine dense + BM25
  scores - normalize both to [0,1] before combining (min-max normalize
  each score list independently), don't average raw scores from
  different scales.

Wire this into toolrouter/__init__.py's ToolRouter facade if it isn't
already connected (check the existing __init__.py first).
```

---

### Prompt 6 — BM25 (do this after Prompt 5 works, not before)

```
Read ARCHITECTURE.md, section "index/bm25.py". Implement BM25Index in
toolrouter/index/bm25.py using the rank_bm25 library. build(ids, texts)
tokenizes with a simple whitespace/lowercase tokenizer and fits BM25Okapi.
search(query, k) returns [(id, score), ...] sorted descending.

This is explicitly optional per BUILD_PLAN.md - only do this once dense
retrieval alone is working and tested.
```

---

### Prompt 7 — confidence gate

```
Read ARCHITECTURE.md, section "router/confidence_gate.py" carefully -
this is the most important function in the whole project, implement it
exactly as specified including both edge cases (fewer than 2
candidates; all scores below an absolute floor).

Implement confidence_gate() in toolrouter/router/confidence_gate.py.

Write tests/test_confidence_gate.py with at least these cases:
1. Large gap between top-1 and top-2 -> returns only min_k results
2. Small gap -> returns up to max_k results
3. Only 1 candidate passed in -> handles gracefully, doesn't crash
4. All scores very low (below floor) -> returns empty list or a
   clearly-flagged "no confident match" result, not a forced guess

After implementing, show me the test output.
```

---

### Prompt 8 — explainability

```
Read ARCHITECTURE.md, section "router/explain.py". Implement
explain_candidates() in toolrouter/router/explain.py.

Keep reasons honest and template-based for v1: cite the actual score and
which query terms overlapped with the tool's name/description. Do NOT
generate claims like "previously used successfully" or "learned from
feedback" - no historical-success tracking exists in this version, so
don't fabricate a reason that implies it does.
```

---

### Prompt 9 — prompt builder + end-to-end quickstart

```
Read ARCHITECTURE.md, section "router/prompt_builder.py". Implement
build_tool_prompt() in toolrouter/router/prompt_builder.py - it should
take the final routed list[Tool] and produce a compact text block
suitable for injecting into an LLM system prompt (name + description +
parameter schema per tool, nothing else).

Then write examples/quickstart.py that:
1. Loads examples/swiggy_manifest.json via ToolRouter.from_manifest()
2. Runs router.route() on 4 hardcoded example queries - at least one
   obviously "clean" query and one deliberately ambiguous one (see
   BENCHMARK.md's category definitions)
3. Prints, for each: the query, the routed tools with scores, the
   explanation, and the final prompt-builder output

Run it and show me the output.
```

---

### Prompt 10 — CommerceBench dataset generation

```
Read BENCHMARK.md in full. Implement bench/generate_dataset.py per its
"Dataset generation" section.

Generate 10-15 queries per tool in examples/swiggy_manifest.json,
covering the "clean", "ambiguous", and "typo" categories described in
BENCHMARK.md. If an LLM API key is available in the environment, use it
to generate natural-sounding queries; if not, fall back to a
template-based generator (e.g. combining tool description keywords into
sentence templates) so the dataset can still be produced offline -
clearly log which mode was used.

Output to bench/dataset.jsonl in the exact format shown in BENCHMARK.md
(one JSON object per line: query, correct_tool, category).
```

---

### Prompt 11 — baselines + evaluation

```
Read BENCHMARK.md's "Baselines" and "Metrics" sections exactly - four
baselines, no more, using the exact metric list given (Top-1/Top-3
accuracy, MRR, Recall@k, prompt tokens, latency, ambiguous-category
behavior).

Implement bench/baselines.py with one function per baseline
(run_all_tools, run_dense, run_hybrid, run_confidence_gate), and
bench/evaluate.py implementing the evaluate() contract from
ARCHITECTURE.md.

Run evaluate() against bench/dataset.jsonl for all four baselines,
write results to bench_results/<baseline>.json and a combined
bench_results/summary.md table, and show me the output.
```

---

### Prompt 12 — final README pass

```
Read the current README.md. Fill in the "Results" table using the real
numbers from bench_results/summary.md - do not invent or round numbers
that aren't actually there. Keep the rest of the README's structure and
tone exactly as it is (Problem / Architecture / Method / Benchmark /
Results / Limitations / Future Work) - don't add marketing language,
don't rename sections, don't add features to the Future Work list that
weren't already there.
```

---

## A note on how to use an AI coding tool well here

Paste one prompt, review the diff, run the tests/example it produces,
*then* move to the next prompt. If you paste all twelve at once, you'll
get a repo-shaped pile of code with no single point where you actually
understood what got built — and the entire point of this project is
that you can defend every design decision in an interview. Going prompt
by prompt is slower but it's the difference between "I built this" and
"an AI built this and I copied it into my resume."
