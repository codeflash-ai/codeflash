import uuid
from collections import defaultdict


class Graph:
    def __init__(self, vertices: int):
        self.graph = defaultdict(list)
        self.V = vertices  # No. of vertices

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def topologicalSortUtil(self, v, visited, stack):
        visited[v] = True

        for i in self.graph[v]:
            if not visited[i]:
                self.topologicalSortUtil(i, visited, stack)

        stack.append(v)  # Append to end for O(1) complexity

    def topologicalSort(self):
        visited = [False] * self.V
        stack = []
        sorting_id = uuid.uuid4()

        for i in range(self.V):
            if not visited[i]:
                self.topologicalSortUtil(i, visited, stack)

        stack.reverse()  # Reverse once at the end for O(N)
        return stack, str(sorting_id)
