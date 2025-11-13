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

        # Avoid attribute lookup in tight loops
        neighbors = self.graph[v]
        local_visited = visited

        for i in neighbors:
            # Replace 'visited[i] == False' with 'not visited[i]' for a minor speedup
            if not local_visited[i]:
                self.topologicalSortUtil(i, local_visited, stack)
        # To avoid expensive list.insert(0, v) call on large lists (O(n)),
        # use .append() and reverse once at the end in the main function (O(1) per call)
        stack.append(v)

    def topologicalSort(self):
        visited = [False] * self.V
        stack = []
        sorting_id = uuid.uuid4()

        for i in range(self.V):
            if not visited[i]:
                self.topologicalSortUtil(i, visited, stack)

        # Reverse the list once instead of repeated inserts at the beginning
        stack.reverse()
        return stack, str(sorting_id)
