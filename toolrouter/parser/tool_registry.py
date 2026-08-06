"""In-memory tool registry.

Dependency-free by design (see ``PROMPTS.md``, Prompt 2): this is a queryable
store for a single process, not a database. It exists so retrieval code can go
from a tool *name* returned by the vector index back to the full :class:`Tool`
object in O(1), and so callers can slice the tool set by server.
"""

from __future__ import annotations

from collections.abc import Iterator

from .manifest_parser import Tool

__all__ = ["ToolRegistry"]


class ToolRegistry:
    """A queryable, insertion-ordered collection of :class:`Tool` objects.

    Examples
    --------
    >>> from toolrouter.parser.manifest_parser import Tool
    >>> registry = ToolRegistry([
    ...     Tool("search_restaurants", "Find places to eat.", {}, "food"),
    ...     Tool("search_products", "Find grocery items.", {}, "instamart"),
    ... ])
    >>> len(registry)
    2
    >>> registry.by_name("search_products").server
    'instamart'
    >>> [t.name for t in registry.by_server("food")]
    ['search_restaurants']
    """

    def __init__(self, tools: list[Tool]) -> None:
        if tools is None:
            raise ValueError("ToolRegistry requires a list of Tool objects, got None.")
        self._tools: list[Tool] = list(tools)
        self._by_name: dict[str, Tool] = {}
        self._by_server: dict[str, list[Tool]] = {}

        for tool in self._tools:
            if not isinstance(tool, Tool):
                raise TypeError(
                    f"ToolRegistry accepts Tool objects only, got {type(tool).__name__}."
                )
            # Last writer wins on duplicates; parse_manifest already rejects them
            # by default, so this only triggers for deliberately-built registries.
            self._by_name[tool.name] = tool
            self._by_server.setdefault(tool.server, []).append(tool)

    # -- core contract ----------------------------------------------------- #
    @property
    def tools(self) -> list[Tool]:
        """All tools, in manifest order. Returns a copy -- callers can't mutate state."""
        return list(self._tools)

    def by_name(self, name: str) -> Tool | None:
        """Look up a tool by exact name. Returns ``None`` when absent."""
        return self._by_name.get(name)

    def by_server(self, server: str) -> list[Tool]:
        """All tools belonging to ``server``. Returns ``[]`` for unknown servers."""
        return list(self._by_server.get(server, []))

    # -- conveniences ------------------------------------------------------ #
    @property
    def servers(self) -> list[str]:
        """Distinct server names, in first-seen order."""
        return list(self._by_server.keys())

    @property
    def names(self) -> list[str]:
        """All tool names, in manifest order."""
        return [tool.name for tool in self._tools]

    def require(self, name: str) -> Tool:
        """Like :meth:`by_name` but raises ``KeyError`` instead of returning ``None``."""
        tool = self._by_name.get(name)
        if tool is None:
            raise KeyError(
                f"Unknown tool {name!r}. Registry holds {len(self._tools)} tools: "
                f"{self.names}"
            )
        return tool

    def embedding_texts(self) -> tuple[list[str], list[str]]:
        """Return ``(ids, texts)`` aligned for index construction."""
        ids = [tool.name for tool in self._tools]
        texts = [tool.to_embedding_text() for tool in self._tools]
        return ids, texts

    # -- dunder ------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name

    def __repr__(self) -> str:
        return (
            f"ToolRegistry(tools={len(self._tools)}, "
            f"servers={self.servers!r})"
        )
