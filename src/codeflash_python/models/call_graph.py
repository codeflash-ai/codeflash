from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash_python.models.models import FunctionSource


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
    forward: dict[FunctionNode, list[CallEdge]] = field(default_factory=dict, init=False, repr=False)
    reverse: dict[FunctionNode, list[CallEdge]] = field(default_factory=dict, init=False, repr=False)
    nodes: set[FunctionNode] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        fwd: dict[FunctionNode, list[CallEdge]] = {}
        rev: dict[FunctionNode, list[CallEdge]] = {}
        nodes: set[FunctionNode] = set()
        for edge in self.edges:
            fwd.setdefault(edge.caller, []).append(edge)
            rev.setdefault(edge.callee, []).append(edge)
            nodes.add(edge.caller)
            nodes.add(edge.callee)
        self.forward = fwd
        self.reverse = rev
        self.nodes = nodes

    def callees_of(self, node: FunctionNode) -> list[CallEdge]:
        return self.forward.get(node, [])

    def callers_of(self, node: FunctionNode) -> list[CallEdge]:
        return self.reverse.get(node, [])

    def descendants(self, node: FunctionNode, max_depth: int | None = None) -> set[FunctionNode]:
        visited: set[FunctionNode] = set()
        forward_map = self.forward
        if max_depth is None:
            queue: deque[FunctionNode] = deque([node])
            while queue:
                current = queue.popleft()
                for edge in forward_map.get(current, []):
                    if edge.callee not in visited:
                        visited.add(edge.callee)
                        queue.append(edge.callee)
        else:
            depth_queue: deque[tuple[FunctionNode, int]] = deque([(node, 0)])
            while depth_queue:
                current, depth = depth_queue.popleft()
                if depth >= max_depth:
                    continue
                for edge in forward_map.get(current, []):
                    if edge.callee not in visited:
                        visited.add(edge.callee)
                        depth_queue.append((edge.callee, depth + 1))
        return visited

    def ancestors(self, node: FunctionNode, max_depth: int | None = None) -> set[FunctionNode]:
        visited: set[FunctionNode] = set()
        reverse_map = self.reverse
        if max_depth is None:
            queue: list[FunctionNode] = [node]
            while queue:
                current = queue.pop()
                for edge in reverse_map.get(current, []):
                    if edge.caller not in visited:
                        visited.add(edge.caller)
                        queue.append(edge.caller)
        else:
            depth_queue: list[tuple[FunctionNode, int]] = [(node, 0)]
            while depth_queue:
                current, depth = depth_queue.pop()
                if depth >= max_depth:
                    continue
                for edge in reverse_map.get(current, []):
                    if edge.caller not in visited:
                        visited.add(edge.caller)
                        depth_queue.append((edge.caller, depth + 1))
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
        all_nodes = self.nodes
        for node in all_nodes:
            in_degree.setdefault(node, 0)
        for edge in self.edges:
            in_degree[edge.callee] = in_degree.get(edge.callee, 0) + 1

        forward_map = self.forward
        queue = deque(node for node, deg in in_degree.items() if deg == 0)
        result: list[FunctionNode] = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for edge in forward_map.get(node, []):
                in_degree[edge.callee] -= 1
                if in_degree[edge.callee] == 0:
                    queue.append(edge.callee)

        if len(result) < len(all_nodes):
            logger.warning(
                "Call graph contains cycles: %d of %d nodes excluded from topological order",
                len(all_nodes) - len(result),
                len(all_nodes),
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

    from codeflash_python.models.models import FunctionSource

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
