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

        # Convert self.graph[v] to a local variable for faster access
        neighbors = self.graph[v]
        for i in neighbors:
            if not visited[i]:
                self.topologicalSortUtil(i, visited, stack)

        # Avoid stack.insert(0, v): expensive O(n) per insert.
        # Instead, use stack.append(v) and reverse once in the topologicalSort method.
        stack.append(v)

    def topologicalSort(self):
        visited = [False] * self.V
        stack = []
        sorting_id = uuid.uuid4()

        for i in range(self.V):
            if not visited[i]:
                self.topologicalSortUtil(i, visited, stack)

        stack.reverse()
        return stack, str(sorting_id)
