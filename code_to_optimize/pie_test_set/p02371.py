def problem_p02371():
    import sys

    readline = sys.stdin.readline

    def dfs(root):

        visited = [False] * n

        queue = [(0, root)]

        longest = (-1, -1)

        while queue:

            total_weight, node = queue.pop()

            if visited[node]:
                continue

            visited[node] = True

            longest = max(longest, (total_weight, node))

            queue.extend((total_weight + w, t) for w, t in edges[node] if not visited[t])

        return longest

    n = int(readline())

    edges = [set() for _ in range(n)]

    for _ in range(n - 1):

        s, t, w = list(map(int, readline().split()))

        edges[s].add((w, t))

        edges[t].add((w, s))

    _, ln = dfs(0)

    ld, _ = dfs(ln)

    print(ld)


problem_p02371()
