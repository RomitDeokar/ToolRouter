"""Tests for retrieval, explanation, prompt building, and the registry."""

from __future__ import annotations

import json

import pytest

from toolrouter.index.bm25 import BM25Index
from toolrouter.index.embed import EmbeddingModel
from toolrouter.index.vector_store import VectorStore
from toolrouter.parser.manifest_parser import Tool
from toolrouter.parser.tool_registry import ToolRegistry
from toolrouter.router.explain import explain_candidates
from toolrouter.router.prompt_builder import (
    build_no_match_prompt,
    build_tool_prompt,
    estimate_tokens,
    render_tool_compact,
    tokens_for_tools,
)
from toolrouter.router.retrieve import (
    Retriever,
    RouteResult,
    ScoredTool,
    _min_max_normalize,
)


# --------------------------------------------------------------------------- #
# ToolRegistry
# --------------------------------------------------------------------------- #
def test_registry_core_contract(sample_tools):
    registry = ToolRegistry(sample_tools)
    assert len(registry) == 3
    assert registry.by_name("book_table").server == "dineout"
    assert registry.by_name("nope") is None
    assert [t.name for t in registry.by_server("food")] == ["search_restaurants"]
    assert registry.by_server("unknown") == []
    assert registry.servers == ["food", "instamart", "dineout"]


def test_registry_tools_property_returns_a_copy(sample_tools):
    registry = ToolRegistry(sample_tools)
    registry.tools.clear()
    assert len(registry) == 3, "internal state must not be mutable via .tools"


def test_registry_require_raises_for_unknown(sample_tools):
    registry = ToolRegistry(sample_tools)
    assert registry.require("book_table").name == "book_table"
    with pytest.raises(KeyError, match="Unknown tool"):
        registry.require("missing")


def test_registry_rejects_non_tool_objects():
    with pytest.raises(TypeError, match="Tool objects only"):
        ToolRegistry(["not a tool"])  # type: ignore[list-item]


def test_registry_rejects_none():
    with pytest.raises(ValueError):
        ToolRegistry(None)  # type: ignore[arg-type]


def test_registry_embedding_texts_align(sample_tools):
    registry = ToolRegistry(sample_tools)
    ids, texts = registry.embedding_texts()
    assert len(ids) == len(texts) == 3
    assert ids == [t.name for t in sample_tools]


def test_registry_iteration_and_membership(sample_tools):
    registry = ToolRegistry(sample_tools)
    assert [t.name for t in registry] == registry.names
    assert "book_table" in registry
    assert "nonexistent" not in registry


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
@pytest.fixture
def retriever(sample_tools) -> Retriever:
    registry = ToolRegistry(sample_tools)
    embedder = EmbeddingModel(force_fallback=True)
    store = VectorStore(dim=embedder.dim)
    ids, texts = registry.embedding_texts()
    store.add(ids, embedder.embed_batch(texts))
    bm25 = BM25Index()
    bm25.build(ids, texts)
    return Retriever(registry, embedder, store, bm25)


def test_retrieve_returns_scored_tools(retriever):
    results = retriever.retrieve("book a table at a restaurant", k=3)
    assert results and all(isinstance(r, ScoredTool) for r in results)
    assert all(r.source == "dense" for r in results)


def test_retrieve_respects_k(retriever):
    assert len(retriever.retrieve("restaurant", k=2)) == 2


def test_retrieve_sorted_descending(retriever):
    scores = [r.score for r in retriever.retrieve("grocery products", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_empty_query_returns_empty(retriever):
    assert retriever.retrieve("", k=5) == []
    assert retriever.retrieve("   ", k=5) == []


def test_retrieve_zero_k_returns_empty(retriever):
    assert retriever.retrieve("anything", k=0) == []


def test_retrieve_rejects_non_string_query(retriever):
    with pytest.raises(TypeError):
        retriever.retrieve(None, k=3)  # type: ignore[arg-type]


def test_retrieve_server_filter(retriever):
    results = retriever.retrieve("search", k=5, server="instamart")
    assert results and all(r.tool.server == "instamart" for r in results)


def test_hybrid_marks_source_and_components(retriever):
    results = retriever.retrieve("book a table", k=3, hybrid=True)
    assert all(r.source == "hybrid" for r in results)
    assert "dense" in results[0].components and "bm25" in results[0].components
    assert "dense_raw" in results[0].components, "raw score needed for the gate's floor"


def test_hybrid_without_bm25_falls_back_to_dense(sample_tools, caplog):
    registry = ToolRegistry(sample_tools)
    embedder = EmbeddingModel(force_fallback=True)
    store = VectorStore(dim=embedder.dim)
    ids, texts = registry.embedding_texts()
    store.add(ids, embedder.embed_batch(texts))
    retriever = Retriever(registry, embedder, store, bm25=None)

    with caplog.at_level("WARNING"):
        results = retriever.retrieve("book a table", k=3, hybrid=True)
    assert all(r.source == "dense" for r in results)
    assert any("no BM25" in record.message for record in caplog.records)


def test_hybrid_scores_within_unit_range(retriever):
    for result in retriever.retrieve("book a restaurant table", k=3, hybrid=True):
        assert 0.0 <= result.score <= 1.0


def test_dense_weight_bounds_validated(retriever):
    with pytest.raises(ValueError, match="dense_weight"):
        Retriever(retriever.registry, retriever.embedder, retriever.vector_store, dense_weight=1.5)


def test_min_max_normalize_behaviour():
    assert _min_max_normalize([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
    assert _min_max_normalize([]) == []
    # A flat non-zero list keeps its signal rather than being zeroed out.
    assert _min_max_normalize([5.0, 5.0]) == [1.0, 1.0]
    assert _min_max_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_empty_registry_retrieves_nothing():
    embedder = EmbeddingModel(force_fallback=True)
    retriever = Retriever(ToolRegistry([]), embedder, VectorStore(dim=embedder.dim))
    assert retriever.retrieve("anything", k=5) == []


# --------------------------------------------------------------------------- #
# explain
# --------------------------------------------------------------------------- #
def _scored(name: str, score: float, **kwargs) -> ScoredTool:
    tool = kwargs.pop(
        "tool",
        Tool(name, f"Description for {name}.", {"properties": {"party_size": {}}}, "dineout"),
    )
    return ScoredTool(tool=tool, score=score, source=kwargs.pop("source", "dense"), **kwargs)


def test_explain_returns_one_dict_per_candidate():
    scored = [_scored("book_table", 0.91), _scored("check_table", 0.80)]
    rows = explain_candidates("book a table", scored)
    assert len(rows) == 2
    assert rows[0]["rank"] == 1 and rows[1]["rank"] == 2
    for row in rows:
        assert set(row) >= {"tool", "score", "reason", "matched_terms", "source"}
        assert row["reason"].endswith(".")


def test_explain_cites_matched_query_terms():
    tool = Tool("book_table", "Reserve a table for dine-in.", {"properties": {"party_size": {}}}, "dineout")
    rows = explain_candidates("book a table", [_scored("book_table", 0.9, tool=tool)])
    assert "book" in rows[0]["matched_terms"]["name"]
    assert "table" in rows[0]["matched_terms"]["name"]
    assert "book" in rows[0]["reason"]


def test_explain_subject_verb_agreement_for_name_matches():
    """The verb must agree with the number of matched terms.

    Pluralising only the noun produced "query term 'table' appear in the tool
    name" for every single-term match -- which is the majority of explanations,
    so the most-read sentence in the output was the ungrammatical one.
    """
    tool = Tool("book_table", "Reserve a spot.", {"properties": {}}, "dineout")

    single = explain_candidates("find a table", [_scored("book_table", 0.9, tool=tool)])
    assert single[0]["matched_terms"]["name"] == ["table"]
    assert "query term 'table' appears in the tool name" in single[0]["reason"]

    plural = explain_candidates("book a table", [_scored("book_table", 0.9, tool=tool)])
    assert len(plural[0]["matched_terms"]["name"]) == 2
    assert "query terms 'book', 'table' appear in the tool name" in plural[0]["reason"]
    assert "appears" not in plural[0]["reason"]


def test_explain_reports_parameter_matches():
    tool = Tool("book_table", "Reserve a table.", {"properties": {"party_size": {}}}, "dineout")
    rows = explain_candidates("what party size", [_scored("book_table", 0.9, tool=tool)])
    assert "party" in rows[0]["matched_terms"]["parameters"]


def test_explain_never_fabricates_history_claims():
    """No historical-success tracking exists -- reasons must not imply otherwise."""
    rows = explain_candidates("book a table", [_scored("book_table", 0.95)])
    forbidden = ["previously", "historical", "success rate", "learned", "feedback", "past usage"]
    reason = rows[0]["reason"].lower()
    for phrase in forbidden:
        assert phrase not in reason, f"fabricated reason detected: {phrase!r}"


def test_explain_handles_no_term_overlap_honestly():
    tool = Tool("xyz_tool", "Completely unrelated wording.", {"properties": {}}, "srv")
    rows = explain_candidates("book a table", [_scored("xyz_tool", 0.6, tool=tool)])
    assert "no literal term overlap" in rows[0]["reason"]


def test_explain_empty_candidates_returns_sentinel_row():
    rows = explain_candidates("anything", [])
    assert len(rows) == 1
    assert rows[0]["tool"] is None
    assert "No tool was selected" in rows[0]["reason"]


def test_explain_attaches_gate_reason():
    gate = {"reason": "Gap 0.4 meets the threshold; narrowing to top-1."}
    rows = explain_candidates("q", [_scored("book_table", 0.9)], gate=gate)
    assert rows[0]["gate_reason"] == gate["reason"]


def test_explain_stopwords_excluded_from_matches():
    tool = Tool("the_tool", "A tool for the thing.", {"properties": {}}, "srv")
    rows = explain_candidates("the a for", [_scored("the_tool", 0.6, tool=tool)])
    assert rows[0]["matched_terms"]["name"] == []


def test_explain_describes_hybrid_components():
    scored = _scored("book_table", 0.9, source="hybrid", components={"dense": 0.9, "bm25": 0.7})
    rows = explain_candidates("book table", [scored])
    assert "dense" in rows[0]["reason"] and "BM25" in rows[0]["reason"]


def test_explain_output_is_json_serialisable():
    rows = explain_candidates("book a table", [_scored("book_table", 0.9)])
    json.dumps(rows)


# --------------------------------------------------------------------------- #
# prompt_builder
# --------------------------------------------------------------------------- #
def test_compact_prompt_includes_names_descriptions_and_params(sample_tools):
    prompt = build_tool_prompt(sample_tools)
    for tool in sample_tools:
        assert tool.name in prompt
        assert tool.description in prompt
    assert "cuisine" in prompt and "party_size" in prompt


def test_compact_prompt_marks_required_params(sample_tools):
    prompt = render_tool_compact(sample_tools[0])
    assert "location: string (required)" in prompt


def test_prompt_renders_enums_and_arrays():
    tool = Tool(
        "t", "A tool.",
        {"properties": {
            "seating": {"type": "string", "enum": ["indoor", "outdoor"]},
            "tags": {"type": "array", "items": {"type": "string"}},
        }},
        "srv",
    )
    prompt = render_tool_compact(tool)
    assert "enum(indoor|outdoor)" in prompt
    assert "array<string>" in prompt


def test_prompt_handles_tool_with_no_params():
    tool = Tool("ping", "Ping the server.", {"properties": {}}, "srv")
    assert "params: none" in render_tool_compact(tool)


def test_json_style_prompt_is_valid_json(sample_tools):
    body = build_tool_prompt(sample_tools, style="json", header=False)
    parsed = json.loads(body)
    assert len(parsed) == 3
    assert parsed[0]["name"] == "search_restaurants"


def test_empty_tool_list_yields_no_match_prompt():
    assert build_tool_prompt([]) == build_no_match_prompt()
    assert "cannot handle this request" in build_no_match_prompt()


def test_invalid_style_rejected(sample_tools):
    with pytest.raises(ValueError, match="style must be"):
        build_tool_prompt(sample_tools, style="yaml")


def test_header_toggle(sample_tools):
    with_header = build_tool_prompt(sample_tools, header=True)
    without_header = build_tool_prompt(sample_tools, header=False)
    assert "pre-selected as" in with_header
    assert "pre-selected as" not in without_header


def test_examples_included_only_when_requested(sample_tools):
    assert "find italian food near me" not in build_tool_prompt(sample_tools)
    assert "find italian food near me" in build_tool_prompt(sample_tools, include_examples=True)


def test_compact_prompt_is_cheaper_than_json(sample_tools):
    """The core claim of the prompt builder -- verify, don't assume."""
    compact = tokens_for_tools(sample_tools, style="compact")
    verbose = tokens_for_tools(sample_tools, style="json")
    assert compact < verbose


def test_routed_subset_costs_fewer_tokens_than_all_tools(sample_tools):
    one = tokens_for_tools(sample_tools[:1], style="compact")
    all_three = tokens_for_tools(sample_tools, style="compact")
    assert one < all_three


def test_estimate_tokens_behaviour():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") >= 1
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


# --------------------------------------------------------------------------- #
# RouteResult
# --------------------------------------------------------------------------- #
def test_route_result_helpers(sample_tools):
    scored = [ScoredTool(sample_tools[0], 0.9, "dense")]
    result = RouteResult(
        query="q", tools=[sample_tools[0]], scored=scored, explanation=[],
        gate={"mode": "confident"},
    )
    assert result.tool_names == ["search_restaurants"]
    assert result.top_tool is sample_tools[0]
    assert result.is_confident and not result.is_ambiguous
    assert not result.no_confident_match
    json.dumps(result.to_dict())


def test_route_result_empty_top_tool_is_none():
    result = RouteResult(query="q", tools=[], scored=[], explanation=[])
    assert result.top_tool is None
