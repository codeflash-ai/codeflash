def problem_p03321():
    from collections import defaultdict, deque

    import sys

    input = sys.stdin.readline

    N, M = list(map(int, input().split()))

    edges = [list(range(N)) for i in range(N)]

    for i in range(N):

        edges[i].remove(i)

    for _ in range(M):

        a, b = list(map(int, input().split()))

        edges[a - 1].remove(b - 1)

        edges[b - 1].remove(a - 1)

    size = defaultdict(lambda: [0, 0])

    color = [-1] * N

    def set_color(root):

        que = deque([root])

        color[root] = 0

        size[root][0] += 1

        while que:

            v = que.pop()

            for nv in edges[v]:

                if color[nv] < 0:

                    c = 1 - color[v]

                    color[nv] = c

                    size[root][c] += 1

                    que.append(nv)

                elif color[nv] == color[v]:

                    print((-1))

                    sys.exit()

    for i in range(N):

        if color[i] < 0:

            set_color(i)

    S = set([0])

    for a, b in list(size.values()):

        S = set(s + b for s in S) | set(s + a for s in S)

    ans = min(x * (x - 1) // 2 + (N - x) * (N - x - 1) // 2 for x in S)

    print(ans)


problem_p03321()
