"""End-to-end tests: the ToolRouter facade and the benchmark harness.

These run against the offline hash embedder (see ``conftest.py``), so they verify
*plumbing* -- that every component connects correctly and the contracts hold.
Retrieval *quality* is measured separately by CommerceBench, and the
``@pytest.mark.semantic`` tests here check quality only when a real model loads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolrouter import ToolRouter
from toolrouter.bench.baselines import BASELINES, run_baseline
from toolrouter.bench.evaluate import evaluate, render_summary
from toolrouter.bench.generate_dataset import (
    generate_dataset,
    load_dataset,
    write_dataset,
)
from toolrouter.bench.metrics import (
    ndcg_at_k,
    percentile,
    rank_of,
    recall_at_k,
    reciprocal_rank,
    top_n_accuracy,
)
from toolrouter.parser.manifest_parser import ManifestError, parse_manifest


# --------------------------------------------------------------------------- #
# Facade construction
# --------------------------------------------------------------------------- #
def test_from_manifest_indexes_every_tool(router):
    assert len(router.registry) == len(router.vector_store)
    assert len(router.registry) > 0


def test_stats_reports_configuration(router):
    stats = router.stats()
    assert stats["tools"] == len(router.registry)
    assert set(stats["servers"]) == {"food", "instamart", "dineout"}
    assert stats["embedder_is_fallback"] is True, "tests must run offline"
    assert stats["bm25"] is not None
    json.dumps(stats)


def test_from_tools_constructor(sample_tools):
    built = ToolRouter.from_tools(sample_tools)
    assert len(built.registry) == 3


def test_from_manifests_combines_servers(write_manifest):
    first = write_manifest(
        {"server": "github", "tools": [{"name": "create_issue", "description": "Open an issue."}]},
        "gh.json",
    )
    second = write_manifest(
        {"server": "slack", "tools": [{"name": "post_message", "description": "Post a message."}]},
        "slack.json",
    )
    combined = ToolRouter.from_manifests([first, second])
    assert combined.registry.servers == ["github", "slack"]
    assert len(combined.registry) == 2


def test_from_manifests_rejects_name_collisions(write_manifest):
    doc = {"server": "a", "tools": [{"name": "dup", "description": "A tool."}]}
    first = write_manifest(doc, "one.json")
    second = write_manifest({**doc, "server": "b"}, "two.json")
    with pytest.raises(ManifestError, match="collide"):
        ToolRouter.from_manifests([first, second])


def test_load_manifest_extends_and_reindexes(router, write_manifest):
    """``load_manifest`` is part of the documented API surface."""
    extra = write_manifest(
        {"server": "extra", "tools": [{"name": "brand_new_tool", "description": "Does a thing."}]}
    )
    before = len(router.registry)
    router.load_manifest(extra)
    assert len(router.registry) == before + 1
    assert len(router.vector_store) == before + 1
    assert router.registry.by_name("brand_new_tool") is not None


def test_load_manifest_rejects_duplicates(router, mock_manifest_path):
    with pytest.raises(ManifestError, match="re-declares"):
        router.load_manifest(mock_manifest_path)


def test_embedder_dim_mismatch_rejected(sample_tools):
    from toolrouter.index.embed import EmbeddingModel
    from toolrouter.index.vector_store import VectorStore
    from toolrouter.parser.tool_registry import ToolRegistry

    embedder = EmbeddingModel(force_fallback=True)
    with pytest.raises(ValueError, match="does not match"):
        ToolRouter(
            ToolRegistry(sample_tools),
            embedder=embedder,
            vector_store=VectorStore(dim=embedder.dim + 1),
        )


def test_requires_a_registry():
    with pytest.raises(TypeError, match="ToolRegistry"):
        ToolRouter(["not a registry"])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# route()
# --------------------------------------------------------------------------- #
def test_route_returns_complete_result(router):
    result = router.route("book a table for four tonight")
    assert result.query
    assert result.gate and "mode" in result.gate
    assert len(result.explanation) == max(len(result.tools), 1)
    assert result.candidates, "pre-gate candidates must be retained for auditing"
    assert result.latency_ms >= 0.0
    json.dumps(result.to_dict())


def test_route_respects_gate_bounds(router):
    result = router.route("book a table", min_k=1, max_k=3)
    assert len(result.tools) <= 3


def test_route_with_gate_disabled_returns_fixed_k(router):
    result = router.route("book a table", k=4, use_gate=False)
    assert len(result.tools) == 4
    assert result.gate["mode"] == "disabled"


def test_route_never_returns_more_than_max_k(router):
    for query in ["book a table", "buy milk", "cancel my order", "xyzzy nonsense"]:
        assert len(router.route(query, max_k=5).tools) <= 5


def test_route_gated_tools_are_a_prefix_of_candidates(router):
    """The gate filters -- it must never reorder or inject tools."""
    result = router.route("order food for delivery")
    if result.tools:
        candidate_names = [c.tool.name for c in result.candidates]
        assert result.tool_names == candidate_names[: len(result.tool_names)]


def test_out_of_domain_query_returns_no_confident_match(router):
    """The floor must reject queries no tool can serve."""
    result = router.route("what is the airspeed velocity of an unladen swallow")
    assert result.no_confident_match or len(result.tools) > 1, (
        "an unservable query must not yield a single confident tool"
    )


def test_explain_convenience_wrapper(router):
    rows = router.explain("book a table tonight")
    assert isinstance(rows, list) and rows


def test_build_prompt_from_query_and_from_result(router):
    from_query = router.build_prompt("book a table tonight")
    from_result = router.build_prompt(router.route("book a table tonight"))
    assert from_query == from_result
    assert isinstance(from_query, str) and from_query


def test_build_prompt_contains_only_routed_tools(router):
    result = router.route("book a table for four tonight")
    prompt = router.build_prompt(result)
    routed = set(result.tool_names)
    for tool in router.registry.tools:
        if tool.name not in routed:
            assert f"- {tool.name}(" not in prompt, f"{tool.name} leaked into the prompt"


def test_routed_prompt_is_smaller_than_all_tools_prompt(router):
    """The central efficiency claim of the whole project."""
    from toolrouter.router.prompt_builder import estimate_tokens

    routed = estimate_tokens(router.build_prompt("book a table for four tonight"))
    everything = estimate_tokens(router.all_tools_prompt())
    assert routed < everything


def test_server_filter_end_to_end(router):
    result = router.route("search for something", server="instamart", use_gate=False, k=3)
    assert all(t.server == "instamart" for t in result.tools)


def test_repr_is_informative(router):
    assert "ToolRouter(" in repr(router)


# --------------------------------------------------------------------------- #
# Semantic behaviour (needs a real embedding model)
# --------------------------------------------------------------------------- #
@pytest.mark.semantic
def test_real_model_routes_obvious_queries_correctly(mock_manifest_path, semantic_embedder):
    from toolrouter.index.vector_store import VectorStore

    real_router = ToolRouter.from_manifest(
        mock_manifest_path,
        embedder=semantic_embedder,
        vector_store=VectorStore(dim=semantic_embedder.dim),
    )
    expectations = {
        "book a table for four people tonight at 8pm": "book_restaurant_table",
        "where is my food order right now": "track_food_delivery",
        "i need to buy milk and bread": "search_grocery_products",
    }
    for query, expected in expectations.items():
        candidates = real_router.retrieve(query, k=3)
        names = [c.tool.name for c in candidates]
        assert expected in names, f"{query!r} -> {names}, expected {expected}"


@pytest.mark.semantic
def test_real_model_gate_narrows_on_clear_query(mock_manifest_path, semantic_embedder):
    from toolrouter.index.vector_store import VectorStore

    real_router = ToolRouter.from_manifest(
        mock_manifest_path,
        embedder=semantic_embedder,
        vector_store=VectorStore(dim=semantic_embedder.dim),
    )
    result = real_router.route("cancel my table reservation for tomorrow")
    assert result.tools
    assert result.tools[0].name == "cancel_table_reservation"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_rank_of():
    assert rank_of(["a", "b", "c"], "b") == 2
    assert rank_of(["a"], "z") is None


def test_top_n_accuracy():
    assert top_n_accuracy(["a", "b", "c"], "c", 3) == 1.0
    assert top_n_accuracy(["a", "b", "c"], "c", 2) == 0.0
    assert top_n_accuracy([], "a", 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], "a") == 1.0
    assert reciprocal_rank(["a", "b"], "b") == 0.5
    assert reciprocal_rank(["a"], "z") == 0.0


def test_recall_equals_top_k_for_single_relevant_item():
    """Documented in BENCHMARK.md -- assert it so the claim stays true."""
    ranked = ["a", "b", "c", "d"]
    for k in (1, 3, 5):
        assert recall_at_k(ranked, "c", k) == top_n_accuracy(ranked, "c", k)


def test_ndcg_decreases_with_rank():
    assert ndcg_at_k(["a", "b"], "a", 5) == 1.0
    assert 0.0 < ndcg_at_k(["b", "a"], "a", 5) < 1.0
    assert ndcg_at_k(["b", "c"], "a", 5) == 0.0


def test_percentile():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert percentile(values, 50) == pytest.approx(2.5)
    assert percentile([], 50) == 0.0
    assert percentile([7.0], 95) == 7.0


def test_percentile_rejects_invalid_q():
    with pytest.raises(ValueError):
        percentile([1.0], 150)


# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #
def test_template_dataset_generation_is_offline_and_labelled(mock_manifest_path):
    rows, meta = generate_dataset(mock_manifest_path, per_tool=10, use_llm=False)
    assert meta["generation_mode"] == "template"
    assert rows
    assert set(meta["by_category"]) <= {"clean", "ambiguous", "typo", "adversarial"}
    tool_names = {t.name for t in parse_manifest(mock_manifest_path)}
    for row in rows:
        assert row.correct_tool in tool_names, "every label must be a real tool"
        assert row.query.strip()


def test_dataset_generation_is_deterministic(mock_manifest_path):
    first, _ = generate_dataset(mock_manifest_path, per_tool=8, seed=7, use_llm=False)
    second, _ = generate_dataset(mock_manifest_path, per_tool=8, seed=7, use_llm=False)
    assert [(r.query, r.correct_tool) for r in first] == [
        (r.query, r.correct_tool) for r in second
    ]


def test_dataset_covers_every_tool(mock_manifest_path):
    rows, _ = generate_dataset(mock_manifest_path, per_tool=10, use_llm=False)
    labelled = {row.correct_tool for row in rows}
    all_tools = {t.name for t in parse_manifest(mock_manifest_path)}
    assert labelled == all_tools, "every tool needs queries or its recall is unmeasured"


def test_ambiguous_queries_are_realistic_phrases(mock_manifest_path):
    """Regression guard: an early version emitted bare keywords like "filtered"."""
    rows, _ = generate_dataset(mock_manifest_path, per_tool=12, use_llm=False)
    ambiguous = [r for r in rows if r.category == "ambiguous"]
    assert ambiguous
    assert all(len(r.query.split()) >= 2 for r in ambiguous), (
        "single-word queries measure tokenisation, not routing"
    )


def test_write_and_load_dataset_roundtrip(mock_manifest_path, tmp_path):
    rows, meta = generate_dataset(mock_manifest_path, per_tool=6, use_llm=False)
    path = tmp_path / "dataset.jsonl"
    write_dataset(rows, str(path), meta=meta)
    assert path.with_suffix(".meta.json").is_file()

    loaded = load_dataset(str(path))
    assert [(r.query, r.correct_tool, r.category) for r in loaded] == [
        (r.query, r.correct_tool, r.category) for r in rows
    ]


def test_load_missing_dataset_raises_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Generate it first"):
        load_dataset(str(tmp_path / "absent.jsonl"))


def test_load_malformed_dataset_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"query": "no label here"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_dataset(str(path))


# --------------------------------------------------------------------------- #
# Baselines and evaluation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("baseline", BASELINES)
def test_every_baseline_runs(router, baseline):
    outcome = run_baseline(baseline, router, "book a table for four", k=5)
    assert outcome.ranked_tools or baseline == "confidence_gate"
    assert outcome.prompt_tokens >= 0
    assert outcome.tools_in_context == len(outcome.ranked_tools)


def test_all_tools_baseline_includes_every_tool(router):
    outcome = run_baseline("all_tools", router, "anything", k=5)
    assert len(outcome.ranked_tools) == len(router.registry)


def test_fixed_topk_baselines_return_exactly_k(router):
    for baseline in ("dense", "hybrid"):
        outcome = run_baseline(baseline, router, "book a table", k=3)
        assert len(outcome.ranked_tools) == 3, baseline


def test_gate_baseline_uses_fewer_tools_than_fixed_topk(router):
    """The gate's efficiency claim, on average across several queries."""
    queries = ["book a table for four", "where is my food", "buy milk", "cancel my booking"]
    gated = sum(len(run_baseline("confidence_gate", router, q, k=5).ranked_tools) for q in queries)
    fixed = sum(len(run_baseline("dense", router, q, k=5).ranked_tools) for q in queries)
    assert gated < fixed


def test_hybrid_baseline_requires_bm25(mock_manifest_path):
    no_bm25 = ToolRouter.from_manifest(mock_manifest_path, use_hybrid=False)
    with pytest.raises(RuntimeError, match="use_hybrid=True"):
        run_baseline("hybrid", no_bm25, "book a table", k=5)


def test_unknown_baseline_rejected(router):
    with pytest.raises(ValueError, match="Unknown baseline"):
        run_baseline("magic", router, "query", k=5)


def test_evaluate_produces_expected_metric_keys(router, mock_manifest_path, tmp_path):
    rows, meta = generate_dataset(mock_manifest_path, per_tool=6, use_llm=False)
    dataset_path = tmp_path / "ds.jsonl"
    write_dataset(rows, str(dataset_path), meta=meta)

    results = evaluate(
        str(dataset_path), router, "dense", k=5, output_dir=str(tmp_path / "out")
    )
    for key in (
        "top_1_accuracy", "top_3_accuracy", "mrr", "ndcg_at_5",
        "recall_at_1", "recall_at_3", "recall_at_5",
        "avg_prompt_tokens", "latency_p50_ms", "latency_p95_ms",
        "by_category", "ambiguous_behaviour", "config",
    ):
        assert key in results, f"missing metric: {key}"
    assert 0.0 <= results["top_1_accuracy"] <= 1.0
    assert results["top_1_accuracy"] <= results["top_3_accuracy"]


def test_evaluate_writes_result_files(router, mock_manifest_path, tmp_path):
    rows, meta = generate_dataset(mock_manifest_path, per_tool=5, use_llm=False)
    dataset_path = tmp_path / "ds.jsonl"
    write_dataset(rows, str(dataset_path), meta=meta)
    out_dir = tmp_path / "results"

    evaluate(str(dataset_path), router, "dense", k=5, output_dir=str(out_dir))
    assert (out_dir / "dense.json").is_file()
    assert (out_dir / "dense_predictions.jsonl").is_file()
    payload = json.loads((out_dir / "dense.json").read_text())
    assert payload["baseline"] == "dense"
    assert payload["config"]["router"]["embedder_is_fallback"] is True


def test_all_tools_baseline_reports_zero_token_reduction(router, mock_manifest_path, tmp_path):
    """It is the reference point, so it must measure 0% against itself."""
    rows, meta = generate_dataset(mock_manifest_path, per_tool=4, use_llm=False)
    dataset_path = tmp_path / "ds.jsonl"
    write_dataset(rows, str(dataset_path), meta=meta)
    results = evaluate(
        str(dataset_path), router, "all_tools", k=5, output_dir=str(tmp_path / "o")
    )
    assert results["token_reduction_vs_all_tools"] == pytest.approx(0.0, abs=1e-9)


def test_evaluate_rejects_unknown_baseline(router, tmp_path):
    with pytest.raises(ValueError, match="Unknown baseline"):
        evaluate("ignored.jsonl", router, "nope", output_dir=str(tmp_path))


def test_render_summary_produces_markdown_tables():
    fake = {
        "dense": {
            "baseline": "dense", "k": 5,
            "top_1_accuracy": 0.9, "top_3_accuracy": 0.95, "mrr": 0.92, "ndcg_at_5": 0.93,
            "recall_at_1": 0.9, "recall_at_3": 0.95, "recall_at_5": 0.97,
            "avg_prompt_tokens": 300.0, "avg_tools_in_context": 5.0,
            "latency_p50_ms": 0.5, "latency_p95_ms": 0.9,
            "token_reduction_vs_all_tools": 0.88,
            "by_category": {"clean": {"queries": 10, "top_1_accuracy": 0.9, "top_3_accuracy": 1.0}},
            "ambiguous_behaviour": {"ambiguous_queries": 0},
            "config": {"router": {"tools": 16}, "dataset": "x.jsonl", "dataset_queries": 10},
        }
    }
    summary = render_summary(fake)
    assert "# CommerceBench Results" in summary
    assert "| Dense only |" in summary
    assert "90.0%" in summary


def test_render_summary_handles_no_results():
    assert "No results" in render_summary({})


# --------------------------------------------------------------------------- #
# Multi-server routing
#
# The README presents from_manifests() as the case the project is really aimed
# at (one agent wired to several MCP servers at once), so it needs coverage
# against the two shipped example manifests rather than only synthetic ones.
# --------------------------------------------------------------------------- #
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
DEVTOOLS_MANIFEST = EXAMPLES / "devtools_manifest.json"


@pytest.fixture(scope="session")
def devtools_manifest_path() -> str:
    if not DEVTOOLS_MANIFEST.is_file():
        pytest.skip(f"Devtools manifest missing at {DEVTOOLS_MANIFEST}")
    return str(DEVTOOLS_MANIFEST)


@pytest.fixture
def multi_router(mock_manifest_path, devtools_manifest_path):
    return ToolRouter.from_manifests(
        [mock_manifest_path, devtools_manifest_path], use_hybrid=True
    )


def test_from_manifests_combines_every_server(multi_router):
    assert len(multi_router.tools) == 26
    assert set(multi_router.registry.servers) == {
        "food", "instamart", "dineout", "github", "slack",
    }


def test_from_manifests_indexes_all_combined_tools(multi_router):
    """Every combined tool must be retrievable, not just the first manifest's."""
    assert len(multi_router.vector_store) == 26
    candidates = multi_router.retrieve("open a pull request", k=26)
    assert {c.tool.name for c in candidates} == {t.name for t in multi_router.tools}


def test_from_manifests_rejects_colliding_tool_names(write_manifest, mock_manifest_path):
    """Silently dropping or shadowing a duplicate tool would be a correctness bug."""
    collide = write_manifest(
        {
            "server": "clone",
            "tools": [
                {
                    "name": "search_restaurants",  # already in the mock manifest
                    "description": "A colliding tool.",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
        name="collide.json",
    )
    with pytest.raises(ManifestError, match="collide"):
        ToolRouter.from_manifests([mock_manifest_path, collide])


@pytest.mark.semantic
def test_cross_server_routing_picks_the_right_domain(
    mock_manifest_path, devtools_manifest_path, semantic_embedder
):
    """With a real model, a dev query must not route into the commerce servers."""
    router = ToolRouter.from_manifests(
        [mock_manifest_path, devtools_manifest_path],
        use_hybrid=True,
        embedder=semantic_embedder,
    )
    assert router.route("raise a PR from my feature branch into main").tools[0].name == (
        "open_pull_request"
    )
    assert router.route("reserve a table for four tonight").tools[0].server == "dineout"
