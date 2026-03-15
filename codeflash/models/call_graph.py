from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.models.models import FunctionSource


class FunctionNode(NamedTuple):
    file_path: Path
    qualified_name: str


@dataclass(frozen=True)
class CalleeMetadata:
    fully_qualified_name: str
    only_function_name: str
    definition_type: str
    source_line: str


@dataclass(frozen=True)
class CallEdge:
    caller: FunctionNode
    callee: FunctionNode
    is_cross_file: bool
    call_count: int | None = None
    total_time_ns: int | None = None
    callee_metadata: CalleeMetadata | None = None


@dataclass
class CallGraph:
    edges: list[CallEdge]
    _forward: dict[FunctionNode, list[CallEdge]] | None = field(default=None, init=False, repr=False)
    _reverse: dict[FunctionNode, list[CallEdge]] | None = field(default=None, init=False, repr=False)
    _nodes: set[FunctionNode] | None = field(default=None, init=False, repr=False)

    def _build_adjacency(self) -> None:
        fwd: dict[FunctionNode, list[CallEdge]] = {}
        rev: dict[FunctionNode, list[CallEdge]] = {}
        nodes: set[FunctionNode] = set()
        for edge in self.edges:
            fwd.setdefault(edge.caller, []).append(edge)
            rev.setdefault(edge.callee, []).append(edge)
            nodes.add(edge.caller)
            nodes.add(edge.callee)
        self._forward = fwd
        self._reverse = rev
        self._nodes = nodes

    @property
    def forward(self) -> dict[FunctionNode, list[CallEdge]]:
        if self._forward is None:
            self._build_adjacency()
        assert self._forward is not None
        return self._forward

    @property
    def reverse(self) -> dict[FunctionNode, list[CallEdge]]:
        if self._reverse is None:
            self._build_adjacency()
        assert self._reverse is not None
        return self._reverse

    @property
    def nodes(self) -> set[FunctionNode]:
        if self._nodes is None:
            self._build_adjacency()
        assert self._nodes is not None
        return self._nodes

    def callees_of(self, node: FunctionNode) -> list[CallEdge]:
        return self.forward.get(node, [])

    def callers_of(self, node: FunctionNode) -> list[CallEdge]:
        return self.reverse.get(node, [])

    def descendants(self, node: FunctionNode, max_depth: int | None = None) -> set[FunctionNode]:
        visited: set[FunctionNode] = set()
        queue: deque[tuple[FunctionNode, int]] = deque([(node, 0)])
        while queue:
            current, depth = queue.popleft()
            if max_depth is not None and depth >= max_depth:
                continue
            for edge in self.callees_of(current):
                if edge.callee not in visited:
                    visited.add(edge.callee)
                    queue.append((edge.callee, depth + 1))
        return visited

    def ancestors(self, node: FunctionNode, max_depth: int | None = None) -> set[FunctionNode]:
        visited: set[FunctionNode] = set()
        reverse_map = self.reverse

        if max_depth is None:
            queue: deque[FunctionNode] = deque([node])
            while queue:
                current = queue.popleft()
                for edge in reverse_map.get(current, []):
                    if edge.caller not in visited:
                        visited.add(edge.caller)
                        queue.append(edge.caller)
        else:
            queue_with_depth: deque[tuple[FunctionNode, int]] = deque([(node, 0)])
            while queue_with_depth:
                current, depth = queue_with_depth.popleft()
                if depth >= max_depth:
                    continue
                for edge in reverse_map.get(current, []):
                    if edge.caller not in visited:
                        visited.add(edge.caller)
                        queue_with_depth.append((edge.caller, depth + 1))

        return visited

    def subgraph(self, nodes: set[FunctionNode]) -> CallGraph:
        filtered = [e for e in self.edges if e.caller in nodes and e.callee in nodes]
        return CallGraph(edges=filtered)

    def leaf_functions(self) -> set[FunctionNode]:
        all_nodes = self.nodes
        return all_nodes - set(self.forward.keys())

    def root_functions(self) -> set[FunctionNode]:
        all_nodes = self.nodes
        return all_nodes - set(self.reverse.keys())

    def topological_order(self) -> list[FunctionNode]:
        in_degree: dict[FunctionNode, int] = {}
        for node in self.nodes:
            in_degree.setdefault(node, 0)
        for edge in self.edges:
            in_degree[edge.callee] = in_degree.get(edge.callee, 0) + 1

        queue = deque(node for node, deg in in_degree.items() if deg == 0)
        result: list[FunctionNode] = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for edge in self.callees_of(node):
                in_degree[edge.callee] -= 1
                if in_degree[edge.callee] == 0:
                    queue.append(edge.callee)

        if len(result) < len(self.nodes):
            logger.warning(
                "Call graph contains cycles: %d of %d nodes excluded from topological order",
                len(self.nodes) - len(result),
                len(self.nodes),
            )

        # Leaves-first: reverse the topological order
        result.reverse()
        return result


def augment_with_trace(graph: CallGraph, trace_db_path: Path) -> CallGraph:
    import sqlite3

    conn = sqlite3.connect(str(trace_db_path))
    try:
        rows = conn.execute(
            "SELECT filename, function, class_name, call_count_nonrecursive, total_time_ns FROM pstats"
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return graph
    conn.close()

    lookup: dict[tuple[str, str], tuple[int, int]] = {}
    for filename, function, class_name, call_count, total_time in rows:
        if class_name:
            qn = f"{class_name}.{function}"
        else:
            qn = function
        lookup[(filename, qn)] = (call_count, total_time)

    augmented_edges: list[CallEdge] = []
    for edge in graph.edges:
        callee_file = str(edge.callee.file_path)
        callee_qn = edge.callee.qualified_name
        stats = lookup.get((callee_file, callee_qn))
        if stats is not None:
            call_count, total_time = stats
            augmented_edges.append(
                CallEdge(
                    caller=edge.caller,
                    callee=edge.callee,
                    is_cross_file=edge.is_cross_file,
                    call_count=call_count,
                    total_time_ns=total_time,
                    callee_metadata=edge.callee_metadata,
                )
            )
        else:
            augmented_edges.append(edge)

    return CallGraph(edges=augmented_edges)


def callees_from_graph(graph: CallGraph) -> tuple[dict[Path, set[FunctionSource]], list[FunctionSource]]:

    from codeflash.models.models import FunctionSource

    file_path_to_function_source: dict[Path, set[FunctionSource]] = defaultdict(set)
    function_source_list: list[FunctionSource] = []

    for edge in graph.edges:
        meta = edge.callee_metadata
        if meta is None:
            continue
        callee_path = edge.callee.file_path
        fs = FunctionSource(
            file_path=callee_path,
            qualified_name=edge.callee.qualified_name,
            fully_qualified_name=meta.fully_qualified_name,
            only_function_name=meta.only_function_name,
            source_code=meta.source_line,
            definition_type=meta.definition_type,
        )
        file_path_to_function_source[callee_path].add(fs)
        function_source_list.append(fs)

    return file_path_to_function_source, function_source_list
