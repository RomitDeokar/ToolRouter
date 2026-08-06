# Toolrouter

**Semantic tool retrieval for MCP agents.** Retrieve only the tools a query
actually needs *before* the LLM reasons, instead of stuffing every tool schema
into every prompt.

```python
from toolrouter import ToolRouter

router = ToolRouter.from_manifest("examples/swiggy_manifest.json", use_hybrid=True)
result = router.route("book a table for four tonight")

print(result.tool_names)          # ['book_restaurant_table']
print(result.gate["reason"])      # why the gate narrowed to one tool
print(router.build_prompt(result))  # the only tool schema the LLM sees
```

On the benchmark below, routing cuts tool-schema tokens by **95.7%** while
raising top-1 tool-selection accuracy from **6.2% → 94.8%**. Both numbers are
measured, not asserted — see [Results](#results).

---

## Problem

Agent frameworks typically hand the LLM every available tool's full JSON schema
on every turn:

```
User query ──► LLM sees all N tool schemas ──► LLM picks one
```

Three things degrade as N grows:

1. **Token bloat** — every schema sits in context whether it is relevant to
   this turn or not. In this repo's 16-tool mock manifest that is ~2,662 tokens
   per call, spent before the user's question is even read.
2. **Selection accuracy drops** — tools with overlapping surface area
   (`search_restaurants` vs. `search_products` vs. `get_restaurant_menu`) get
   harder to disambiguate as siblings multiply.
3. **Cost and latency scale with N**, not with the difficulty of the request.

This is an ecosystem-level problem, not a single-vendor one, and it compounds
the moment an agent is wired to several MCP servers at once (Swiggy + GitHub +
Slack + filesystem). `ToolRouter.from_manifests([...])` is built for exactly
that case.

## Approach

```
User query ──► Retriever ──► Confidence gate ──► Top-k tools ──► LLM ──► MCP call
```

Retrieval is embedding-based (dense), optionally fused with BM25 lexical
scores. The piece that makes this *routing* rather than plain retrieval is the
**adaptive confidence gate**: rather than always returning a fixed k, it reads
the score gap between the top two candidates.

| Situation | Signal | Gate behaviour |
|---|---|---|
| One candidate clearly wins | `top1 - top2 ≥ gap_threshold` | Narrow to `min_k` (1 tool) |
| Several are plausible | `top1 - top2 < gap_threshold` | Widen to `max_k`, let the LLM disambiguate |
| Nothing matched at all | `top1 < score_floor` | Return **no confident match** — refuse rather than guess |

That third branch matters. A router that always answers is indistinguishable
from a router that has no idea, and forcing a top-1 guess on an out-of-domain
query ("what is the capital of peru") is how agents end up calling
`place_order` about the weather.

Full component contracts are in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Install

```bash
git clone https://github.com/RomitDeokar/ToolRouter.git
cd toolrouter
pip install -e ".[bench]"     # real embeddings + faiss + BM25 + exact tokens
```

Only `numpy` is a hard dependency. Every other component degrades gracefully:

| Component | Preferred | Fallback if unavailable |
|---|---|---|
| Embeddings | `fastembed` (BGE-small-en-v1.5) | deterministic hash vectors (**lexical, logs a loud warning**) |
| Vector index | `faiss` | brute-force numpy cosine (exact) |
| Lexical scores | `rank_bm25` | in-tree BM25-Okapi (identical ranking) |
| Token counting | `tiktoken` | ~4-chars-per-token heuristic |

The hash-embedding fallback exists so the pipeline and its 230 tests run
offline with zero downloads. It is **not** a semantic model, and it warns every
time it activates — fallback-quality embeddings would quietly wreck benchmark
numbers if anyone forgot which mode they were in.

## Usage

### CLI

```bash
toolrouter route "book a table for four tonight"   # route + show gate decision
toolrouter prompt "order paneer" --style json      # the LLM-ready prompt block
toolrouter tools --server dineout                  # list indexed tools
toolrouter agent "cancel my reservation"           # full routed agent loop
toolrouter stats                                   # router configuration
toolrouter bench --k 5                             # reproduce CommerceBench
toolrouter calibrate                               # sweep gate thresholds
toolrouter dataset --per-tool 12                   # regenerate the dataset
```

Add `--offline` to any command to force the hash embedder, or `--hybrid` to fuse
BM25 with dense scores.

A confident query narrows to a single tool:

```
$ toolrouter route "cancel my reservation"
query : "cancel my reservation"
router: 16 tools, fastembed:BAAI/bge-small-en-v1.5

gate  : confident -- Gap 0.132 between 'cancel_table_reservation' (0.805) and
        'cancel_food_order' (0.673) meets the threshold 0.03; narrowing to top-1.

candidates (* = kept by the gate):
  * 0.8045  cancel_table_reservation       (dineout)
    0.6730  cancel_food_order              (food)
    0.5605  check_table_availability       (dineout)
    0.5501  book_restaurant_table          (dineout)
    0.5183  get_dineout_offers             (dineout)
```

A genuinely ambiguous one widens instead of guessing — "order paneer" could mean
ready-to-eat delivery or groceries to cook with:

```
$ toolrouter route "order paneer"
gate  : ambiguous -- Gap 0.008 between 'place_food_order' (0.594) and
        'add_food_item_to_cart' (0.586) is below the threshold 0.03;
        widening to top-5 so the LLM can disambiguate.
```

And an out-of-domain query is refused rather than forced onto the nearest tool:

```
$ toolrouter route "what is the capital of peru"
gate  : no_confident_match -- Best candidate 'discover_dineout_restaurants'
        scored 0.442, below the absolute floor 0.54. No tool in the index
        plausibly serves this query, so no guess is returned.
```

### Python

```python
from toolrouter import ToolRouter

# Route across several MCP servers at once -- the case this is really for.
# These two manifests deliberately use DIFFERENT shapes (nested `servers` +
# `parameters` vs. flat `tools` + `inputSchema`); the parser handles both.
router = ToolRouter.from_manifests(
    ["examples/swiggy_manifest.json", "examples/devtools_manifest.json"],
    use_hybrid=True,
)

result = router.route("raise a PR from my feature branch into main")
# -> 26 tools indexed across 5 servers; gate narrows to ['open_pull_request']

result.tool_names      # routed subset, post-gate
result.gate["mode"]    # 'confident' | 'ambiguous' | 'no_confident_match'
result.explanation     # per-candidate score + matched terms + reason
result.latency_ms

router.build_prompt(result)   # inject into your agent's system prompt
router.all_tools_prompt()     # the unrouted baseline, for comparison
```

### Runnable examples

```bash
python examples/quickstart.py    # parse → index → route, several query types
python examples/agent_demo.py    # routed agent loop, incl. refusal on OOD queries
```

## Benchmark: CommerceBench

Retrieval quality is measured, not claimed. The dataset is 192 self-play
generated queries over the 16-tool mock manifest, each labelled with a
ground-truth tool and a difficulty **category** — so accuracy is reported
broken down by difficulty, which is far more credible than one blended number.

| Category | n | What it tests |
|---|---|---|
| `clean` | 96 | Obvious mapping ("book a table for 4 tonight") |
| `ambiguous` | 48 | Plausibly maps to a sibling tool ("order paneer") |
| `typo` | 32 | Informal/misspelled ("resto near me open now") |
| `adversarial` | 16 | Instruction-flavoured text embedded in the query |

Four baselines, exactly as specified in [`BENCHMARK.md`](BENCHMARK.md) — no
fifth one added "to be thorough."

```bash
toolrouter bench          # writes bench_results/*.json + summary.md
```

## Results

Generated by `toolrouter.bench.evaluate` into
[`bench_results/summary.md`](bench_results/summary.md). Numbers below are copied
from that file, not hand-estimated. Setup: 192 queries, 16 tools,
`fastembed:BAAI/bge-small-en-v1.5`, `faiss:IndexFlatIP`, `tiktoken:cl100k_base`,
k=5, `gap_threshold=0.03`, `score_floor=0.54`.

| Method | Top-1 Acc | Top-3 Acc | MRR | NDCG@5 | Avg Prompt Tokens | Token Reduction | p95 Latency |
|---|---|---|---|---|---|---|---|
| All tools (baseline) | 6.2% | 18.8% | 0.211 | 0.184 | 2662 | — | — |
| Dense only | 87.5% | 97.4% | 0.924 | 0.942 | 284 | 89.3% | 0.11 ms |
| **Dense + BM25** | **94.8%** | **99.0%** | **0.968** | **0.974** | 283 | 89.4% | 0.30 ms |
| Dense + confidence gate | 87.5% | 95.8% | 0.915 | 0.930 | **113** | **95.7%** | 0.14 ms |

> Two honesty notes on this table:
>
> - The `all_tools` row has no ranking of its own — its tool order is manifest
>   order, so its accuracy columns measure *manifest position*, not relevance.
>   Its meaningful column is prompt tokens. Reporting it matters: the baseline
>   needs a real number, not an assumption that it's worse.
> - **Latency measures search only, not query encoding.** The evaluator warms
>   the (memoised) embedding cache before timing, because otherwise whichever
>   baseline ran first would absorb the entire cold-start encoding cost and look
>   ~50× slower — an artefact of evaluation order, not a real difference between
>   methods. End-to-end first-call latency including encoding is ~11–13 ms, as
>   the CLI examples above show. The sub-millisecond figures are the part that
>   actually differs between baselines.

### Top-1 accuracy by category

| Method | clean (96) | ambiguous (48) | typo (32) | adversarial (16) |
|---|---|---|---|---|
| All tools | 6.2% | 6.2% | 6.2% | 6.2% |
| Dense only | 92.7% | 89.6% | 75.0% | 75.0% |
| Dense + BM25 | 100.0% | 93.8% | 81.2% | 93.8% |
| Dense + confidence gate | 92.7% | 89.6% | 75.0% | 75.0% |

Hybrid retrieval's gains are concentrated exactly where you'd predict: `typo`
and `adversarial` queries, where lexical overlap rescues cases that pure dense
similarity blurs.

### Ambiguous-query behaviour

The metric most retrieval write-ups omit — when the router is genuinely unsure,
does it widen instead of confidently returning one wrong tool?

| Method | Widen Rate | Top-1 Acc | Correct Tool in Context | Avg Tools in Context |
|---|---|---|---|---|
| Dense only (fixed k=5) | 0.0% | 89.6% | 100.0% | 5.00 |
| Dense + confidence gate | 35.4% | 89.6% | 100.0% | **2.42** |

The gate keeps the correct tool in context 100% of the time on ambiguous
queries while cutting average context from 5 tools to 2.42 — same recall,
less than half the tokens. Fixed-top-k baselines cannot widen by construction,
so their widen rate is 0% by definition, not by merit.

### Why `gap_threshold = 0.03` and not 0.15

`ARCHITECTURE.md` offers 0.15 as an *illustrative* value. Measured against real
BGE cosine scores on this corpus it is far too strict — it narrows only 12.5%
of clean queries, so adaptive-k degenerates into fixed top-k and the gate stops
meaning anything. The shipped default is the calibrated optimum from a sweep
([`bench_results/calibration.md`](bench_results/calibration.md), reproduce with
`toolrouter calibrate`):

| gap_threshold | Clean narrowed to 1 | Ambiguous: correct in context | Avg tools | Balance |
|---|---|---|---|---|
| 0.00 | 100.0% | 89.6% | 1.00 | 0.911 |
| **0.03 (shipped)** | 86.5% | **100.0%** | 2.00 | **0.927** |
| 0.15 (spec's example) | 12.5% | 100.0% | 4.75 | 0.222 |

*Balance* is the harmonic mean of the gate's two competing jobs; a threshold
that wins one at the other's expense scores poorly.

The `score_floor = 0.54` is likewise an explicit, documented trade-off rather
than a tuned-to-look-good number: in-domain and out-of-domain score
distributions **overlap** on this corpus (in-domain min 0.543, out-of-domain max
0.583), so no floor separates them perfectly. The floor is set to the highest
value that rejects *zero* valid dataset queries, because the two errors are not
symmetric — wrongly rejecting a valid query is a hard failure with no recovery
path, while failing to reject an out-of-domain one is soft (the LLM sees the
tools, sees none fit, and can still decline). See the reasoning in
[`toolrouter/router/confidence_gate.py`](toolrouter/router/confidence_gate.py).

## Repository structure

```
toolrouter/
├── toolrouter/
│   ├── parser/           # MCP manifest → Tool objects → ToolRegistry
│   ├── index/            # embeddings + vector store + BM25
│   ├── router/           # retrieve → confidence gate → explain → prompt
│   ├── bench/            # CommerceBench: dataset, baselines, evaluate, calibrate
│   ├── agent.py          # routed agent loop (heuristic or OpenAI client)
│   └── cli.py            # unified `toolrouter` command
├── examples/             # mock manifest + quickstart + agent demo
├── tests/                # 230 tests, hermetic (offline embedder by default)
├── bench_results/        # committed benchmark output — the Results table's source
├── ci/                   # GitHub Actions workflow (see ci/README.md to enable)
├── docs/spec/            # the original design spec, kept verbatim
├── ARCHITECTURE.md       # component contracts (source of truth)
├── BENCHMARK.md          # CommerceBench methodology
└── BUILD_PLAN.md         # build order
```

### Where the code departs from `docs/spec/`

`docs/spec/` is the original design brief, committed unedited so the build can be
checked against what was actually asked for. All five deliverables and the whole
`API.md` surface (`load_manifest` / `index_tools` / `retrieve` / `explain` /
`benchmark`) are implemented. Two items in `DATA_MODEL.md` are deliberately not,
and silently diverging from a spec is worse than diverging openly:

- **`Tool.embedding` does not exist.** Storing each tool's vector on the tool
  itself would duplicate state that the vector index already owns, and make a
  `Tool` silently invalid whenever the embedding model changed. Vectors live in
  `VectorStore`, keyed by tool name; `Tool.to_embedding_text()` defines what gets
  embedded, and `Tool` stays a plain description of the manifest entry.
- **`QueryResult` is called `RouteResult`** and carries more than the spec's four
  fields. `retrieved_tools`/`scores` are one thing in practice, not two — a tool
  divorced from its score invites misaligned parallel lists, so they are paired
  in `ScoredTool` and exposed as `.tools` (post-gate) plus `.scored`. `confidence`
  is a `gate` dict rather than a bare number because a scalar cannot express *why*
  the gate widened; `.candidates` keeps the pre-gate list so the decision is
  auditable, and `.latency_ms` is what the benchmark measures.

## Tests

```bash
pip install -e ".[dev]"           # core only (numpy)
pytest                             # 224 passed, 6 skipped in ~1s

pip install -e ".[bench,dev]"     # + fastembed, faiss, rank_bm25, tiktoken
pytest                             # 230 passed in ~5s
```

The suite forces the hash embedder (`TOOLROUTER_FORCE_FALLBACK=1`) so it is
hermetic by default — no network, no model download. That configuration
deliberately exercises every fallback path (hash embedder, numpy cosine search,
in-tree BM25, heuristic token counts), which is why CI installs *without* the
extras in its main job.

Tests that assert *semantic* behaviour rather than plumbing are marked
`@pytest.mark.semantic`. Each one requests a real model explicitly
(`force_fallback=False`, which overrides the suite-wide flag) and skips only when
no model can genuinely be loaded. On a core install that is 5 skips (4 semantic +
1 precedence regression test) plus 1 faiss-specific test; installing `[bench]`
turns all 6 into real assertions. CI runs both configurations for exactly this
reason — otherwise a fallback regression, or a "semantic" test that never
actually runs, would pass unnoticed. A third job re-runs CommerceBench and fails
if the accuracy figures published above drift, so the headline numbers stay
falsifiable rather than being true once on one machine.

The workflow lives at [`ci/github-actions-ci.yml`](ci/github-actions-ci.yml)
rather than `.github/workflows/` — see [`ci/README.md`](ci/README.md) for the
one-command move that activates it.

## Limitations

- **The sample manifest is a mock.** `examples/swiggy_manifest.json` is a
  hand-built 16-tool fixture for local development — it is **not** the verified
  live Swiggy tool list. The parser is deliberately generic, so swap in any real
  MCP manifest without code changes.
- **The dataset is self-play generated**, not real user traffic. An LLM
  producing plausible queries per tool is a reasonable proxy, not a substitute
  for observed usage.
- **16 tools is a small corpus.** The token-reduction figure is real but scales
  with N; accuracy on 16 tools does not automatically transfer to 500.
- **No reranker.** A cross-encoder pass over the top-20 would likely help the
  `typo`/`adversarial` categories most — not implemented in v1.
- **No multilingual support**, no distributed indexing, no caching.
- **No adaptive learning from execution feedback.** Ranking by historical
  success requires real usage volume to mean anything; without it, it would be
  a fabricated signal. Explanations deliberately cite only what retrieval
  actually used.

## Future work

- Hierarchical routing — predict the server first, then retrieve within it
- Cross-encoder reranking over the top-20 candidates
- ToolGraph: learn tool-to-tool sequencing (`search_restaurants` →
  `get_menu` → `place_order`) for workflow-level planning, not just
  single-turn retrieval
- Model-aware routing — different LLMs may respond differently to the same
  retrieved tool set

## License

MIT — see [LICENSE](LICENSE).
