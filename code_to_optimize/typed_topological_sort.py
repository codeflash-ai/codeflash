from collections import defaultdict


class Graph:
    def __init__(self, vertices: int) -> None:
        self.graph: dict[int, list[int]] = defaultdict(list)
        self.V: int = vertices  # No. of vertices

    def addEdge(self, u: int, v: int) -> None:
        self.graph[u].append(v)

    def topologicalSortUtil(self, v: int, visited: list[bool], stack: list[int]) -> None:
        visited[v] = True

        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        stack.insert(0, v)

    def topologicalSort(self) -> list[int]:
        visited: list[bool] = [False] * self.V
        stack: list[int] = []

        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        return stack
