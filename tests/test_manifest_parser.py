"""Tests for the manifest parser -- the component that must be genuinely generic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolrouter.parser.manifest_parser import (
    ManifestError,
    Tool,
    parse_manifest,
    parse_manifest_dict,
)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_valid_manifest_parses(minimal_manifest_dict, write_manifest):
    tools = parse_manifest(write_manifest(minimal_manifest_dict))
    assert [t.name for t in tools] == ["search_restaurants", "place_order"]
    assert all(isinstance(t, Tool) for t in tools)
    assert all(t.server == "food" for t in tools), "root-level server must be inherited"


def test_mock_manifest_parses_all_three_servers(mock_manifest_path):
    tools = parse_manifest(mock_manifest_path)
    assert len(tools) >= 8, "mock manifest should cover a realistic tool count"
    assert {t.server for t in tools} == {"food", "instamart", "dineout"}
    assert len({t.name for t in tools}) == len(tools), "names must be unique"
    for tool in tools:
        assert tool.description, f"{tool.name} has an empty description"
        assert isinstance(tool.parameters, dict)


def test_parse_from_bare_list_of_tools():
    document = [
        {"name": "a", "description": "First tool.", "parameters": {}, "server": "s1"},
        {"name": "b", "description": "Second tool.", "parameters": {}, "server": "s2"},
    ]
    tools = parse_manifest_dict(document)
    assert [(t.name, t.server) for t in tools] == [("a", "s1"), ("b", "s2")]


def test_parse_list_of_server_documents():
    document = [
        {"server": "github", "tools": [{"name": "create_issue", "description": "Open an issue."}]},
        {"server": "slack", "tools": [{"name": "post_message", "description": "Post a message."}]},
    ]
    tools = parse_manifest_dict(document)
    assert [(t.name, t.server) for t in tools] == [
        ("create_issue", "github"),
        ("post_message", "slack"),
    ]


def test_parse_servers_mapping_shape():
    document = {
        "servers": {
            "food": {"tools": [{"name": "x", "description": "X tool."}]},
            "dineout": {"tools": [{"name": "y", "description": "Y tool."}]},
        }
    }
    tools = parse_manifest_dict(document)
    assert {(t.name, t.server) for t in tools} == {("x", "food"), ("y", "dineout")}


def test_tool_name_keyed_mapping_shape():
    document = {
        "server": "fs",
        "tools": {
            "read_file": {"description": "Read a file."},
            "write_file": {"description": "Write a file."},
        },
    }
    tools = parse_manifest_dict(document)
    assert sorted(t.name for t in tools) == ["read_file", "write_file"]


# --------------------------------------------------------------------------- #
# Genericity: field aliases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("params_key", ["parameters", "inputSchema", "input_schema", "schema"])
def test_parameter_schema_aliases(params_key):
    """Real MCP servers spell the schema field differently -- all must work."""
    document = {
        "server": "s",
        "tools": [
            {
                "name": "t",
                "description": "A tool.",
                params_key: {"type": "object", "properties": {"alpha": {"type": "string"}}},
            }
        ],
    }
    tools = parse_manifest_dict(document)
    assert tools[0].parameter_names == ["alpha"]


@pytest.mark.parametrize("tools_key", ["tools", "functions", "actions", "capabilities"])
def test_tool_list_aliases(tools_key):
    document = {"server": "s", tools_key: [{"name": "t", "description": "A tool."}]}
    assert parse_manifest_dict(document)[0].name == "t"


@pytest.mark.parametrize("name_key", ["name", "tool_name", "toolName", "id"])
def test_name_aliases(name_key):
    document = {"server": "s", "tools": [{name_key: "t", "description": "A tool."}]}
    assert parse_manifest_dict(document)[0].name == "t"


@pytest.mark.parametrize("desc_key", ["description", "desc", "summary", "doc"])
def test_description_aliases(desc_key):
    document = {"server": "s", "tools": [{"name": "t", desc_key: "A tool."}]}
    assert parse_manifest_dict(document)[0].description == "A tool."


def test_no_swiggy_specific_field_names_hardcoded():
    """A manifest from an unrelated domain must parse just as well."""
    document = {
        "server": "kubernetes",
        "tools": [
            {
                "name": "scale_deployment",
                "description": "Scale a Kubernetes deployment to N replicas.",
                "inputSchema": {"properties": {"replicas": {"type": "integer"}}},
            }
        ],
    }
    tool = parse_manifest_dict(document)[0]
    assert tool.name == "scale_deployment"
    assert tool.server == "kubernetes"
    assert "replicas" in tool.to_embedding_text()


def test_bare_property_map_gets_schema_envelope():
    """``{"cuisine": {...}}`` without the object/properties envelope still works."""
    document = {
        "server": "s",
        "tools": [
            {"name": "t", "description": "A tool.", "parameters": {"cuisine": {"type": "string"}}}
        ],
    }
    tool = parse_manifest_dict(document)[0]
    assert tool.parameters["type"] == "object"
    assert tool.parameter_names == ["cuisine"]


# --------------------------------------------------------------------------- #
# Errors: clear, specific, never a bare KeyError
# --------------------------------------------------------------------------- #
def test_missing_name_raises_clear_error():
    document = {"server": "s", "tools": [{"description": "No name here."}]}
    with pytest.raises(ManifestError, match="missing a required 'name'"):
        parse_manifest_dict(document)


def test_missing_description_raises_clear_error():
    document = {"server": "s", "tools": [{"name": "t"}]}
    with pytest.raises(ManifestError, match="missing a required 'description'"):
        parse_manifest_dict(document)


def test_missing_required_field_is_not_a_bare_keyerror():
    """The contract: a specific ManifestError, not a raw KeyError/TypeError."""
    with pytest.raises(ManifestError):
        parse_manifest_dict({"server": "s", "tools": [{"description": "x"}]})


def test_empty_name_rejected():
    document = {"server": "s", "tools": [{"name": "   ", "description": "A tool."}]}
    with pytest.raises(ManifestError, match="non-empty string"):
        parse_manifest_dict(document)


def test_no_tool_list_raises():
    with pytest.raises(ManifestError, match="no tool list"):
        parse_manifest_dict({"server": "s", "metadata": {}})


def test_empty_tool_list_raises():
    with pytest.raises(ManifestError, match="zero tools"):
        parse_manifest_dict({"server": "s", "tools": []})


def test_duplicate_tool_names_rejected_by_default():
    document = {
        "server": "s",
        "tools": [
            {"name": "dup", "description": "First."},
            {"name": "dup", "description": "Second."},
        ],
    }
    with pytest.raises(ManifestError, match="Duplicate tool name"):
        parse_manifest_dict(document)


def test_duplicate_tool_names_allowed_when_requested():
    document = {
        "server": "s",
        "tools": [
            {"name": "dup", "description": "First."},
            {"name": "dup", "description": "Second."},
        ],
    }
    assert len(parse_manifest_dict(document, allow_duplicate_names=True)) == 2


def test_bad_parameter_type_raises():
    document = {
        "server": "s",
        "tools": [{"name": "t", "description": "A tool.", "parameters": "not-a-schema"}],
    }
    with pytest.raises(ManifestError, match="must be an object"):
        parse_manifest_dict(document)


def test_missing_file_raises_clear_error():
    with pytest.raises(ManifestError, match="not found"):
        parse_manifest("/nonexistent/path/to/manifest.json")


def test_invalid_json_raises_clear_error(tmp_path):
    target = tmp_path / "broken.json"
    target.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ManifestError, match="not valid JSON"):
        parse_manifest(str(target))


def test_empty_location_raises():
    with pytest.raises(ManifestError, match="non-empty string"):
        parse_manifest("   ")


def test_scalar_document_raises():
    with pytest.raises(ManifestError, match="must be a JSON object or array"):
        parse_manifest_dict(42)


# --------------------------------------------------------------------------- #
# Defaults and optional fields
# --------------------------------------------------------------------------- #
def test_missing_parameters_is_legal():
    """Zero-argument tools exist and must not be rejected."""
    document = {"server": "s", "tools": [{"name": "ping", "description": "Ping."}]}
    tool = parse_manifest_dict(document)[0]
    assert tool.parameters == {"type": "object", "properties": {}}
    assert tool.parameter_names == []


def test_default_server_applied_when_absent():
    document = {"tools": [{"name": "t", "description": "A tool."}]}
    assert parse_manifest_dict(document, default_server="mcp")[0].server == "mcp"


def test_per_tool_server_overrides_root():
    document = {
        "server": "root",
        "tools": [
            {"name": "a", "description": "A.", "server": "specific"},
            {"name": "b", "description": "B."},
        ],
    }
    tools = {t.name: t.server for t in parse_manifest_dict(document)}
    assert tools == {"a": "specific", "b": "root"}


def test_examples_coerced_from_various_shapes():
    document = {
        "server": "s",
        "tools": [
            {"name": "a", "description": "A.", "examples": "single string"},
            {"name": "b", "description": "B.", "examples": ["one", "two"]},
            {"name": "c", "description": "C.", "examples": [{"query": "from object"}]},
        ],
    }
    tools = {t.name: t.examples for t in parse_manifest_dict(document)}
    assert tools["a"] == ["single string"]
    assert tools["b"] == ["one", "two"]
    assert tools["c"] == ["from object"]


# --------------------------------------------------------------------------- #
# to_embedding_text: the parameter-names requirement
# --------------------------------------------------------------------------- #
def test_embedding_text_includes_parameter_names_not_just_description():
    """Per the design spec, parameter names carry disambiguating signal."""
    tool = Tool(
        name="search_restaurants",
        description="Search restaurants by cuisine.",
        parameters={
            "type": "object",
            "properties": {"cuisine": {"type": "string"}, "min_rating": {"type": "number"}},
        },
        server="food",
    )
    text = tool.to_embedding_text()
    assert "search_restaurants" in text
    assert "Search restaurants by cuisine." in text
    assert "cuisine" in text
    assert "min_rating" in text, "parameter names must appear in the embedding text"


def test_embedding_text_disambiguates_similar_descriptions():
    """Two tools with near-identical descriptions must differ once params are in."""
    shared = "Search the catalogue for matching entries."
    restaurants = Tool(
        "search_restaurants", shared,
        {"properties": {"cuisine": {}, "location": {}}}, "food",
    )
    products = Tool(
        "search_products", shared,
        {"properties": {"brand": {}, "category": {}}}, "instamart",
    )
    assert restaurants.to_embedding_text() != products.to_embedding_text()
    assert "cuisine" in restaurants.to_embedding_text()
    assert "brand" in products.to_embedding_text()


def test_embedding_text_includes_humanized_name():
    """``book_restaurant_table`` should also read as natural words."""
    tool = Tool("book_restaurant_table", "Reserve a table.", {}, "dineout")
    assert "book restaurant table" in tool.to_embedding_text()


def test_embedding_text_includes_examples_when_present():
    tool = Tool("t", "A tool.", {}, "s", examples=["do the thing"])
    assert "do the thing" in tool.to_embedding_text()


def test_embedding_text_handles_no_parameters():
    tool = Tool("ping", "Ping the server.", {"properties": {}}, "s")
    assert tool.to_embedding_text().startswith("ping")


# --------------------------------------------------------------------------- #
# Tool helpers
# --------------------------------------------------------------------------- #
def test_required_parameters_exposed():
    tool = Tool(
        "t", "A tool.",
        {"properties": {"a": {}, "b": {}}, "required": ["a"]}, "s",
    )
    assert tool.required_parameters == ["a"]
    assert tool.parameter_names == ["a", "b"]


def test_qualified_name():
    assert Tool("t", "d", {}, "food").qualified_name == "food.t"


def test_to_dict_is_json_serialisable(sample_tools):
    json.dumps([t.to_dict() for t in sample_tools])


# --------------------------------------------------------------------------- #
# The second committed example manifest
#
# devtools_manifest.json exists to prove genericity with a *shipped* fixture,
# not just synthetic in-test dicts: it uses a flat top-level "tools" array with
# a per-tool "server" field and MCP's "inputSchema" spelling, whereas
# swiggy_manifest.json nests tools under a "servers" mapping and spells the
# schema "parameters". Both must parse identically well.
# --------------------------------------------------------------------------- #
DEVTOOLS_MANIFEST = (
    Path(__file__).resolve().parent.parent / "examples" / "devtools_manifest.json"
)


@pytest.fixture(scope="session")
def devtools_manifest_path() -> str:
    if not DEVTOOLS_MANIFEST.is_file():
        pytest.skip(f"Devtools manifest missing at {DEVTOOLS_MANIFEST}")
    return str(DEVTOOLS_MANIFEST)


def test_devtools_manifest_parses_flat_shape(devtools_manifest_path):
    tools = parse_manifest(devtools_manifest_path)
    assert len(tools) == 10
    assert {t.server for t in tools} == {"github", "slack"}


def test_devtools_manifest_input_schema_becomes_parameters(devtools_manifest_path):
    tools = {t.name: t for t in parse_manifest(devtools_manifest_path)}
    # "inputSchema" in the file must surface as .parameters, with properties intact.
    assert tools["open_pull_request"].parameter_names == [
        "repo", "head", "base", "title", "body", "draft",
    ]
    assert tools["open_pull_request"].required_parameters == [
        "repo", "head", "base", "title",
    ]


def test_devtools_manifest_is_labelled_as_mock(devtools_manifest_path):
    """A fictional manifest must say so, so nobody ships it as real data."""
    document = json.loads(Path(devtools_manifest_path).read_text(encoding="utf-8"))
    assert document["_mock"] is True
    assert "FICTIONAL" in document["_comment"]


def test_both_example_manifests_have_disjoint_tool_names(
    mock_manifest_path, devtools_manifest_path
):
    """They are combined in the README's multi-server example, so they must not collide."""
    a = {t.name for t in parse_manifest(mock_manifest_path)}
    b = {t.name for t in parse_manifest(devtools_manifest_path)}
    assert not (a & b)
