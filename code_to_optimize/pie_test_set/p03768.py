def problem_p03768():
    import sys

    input = sys.stdin.readline

    N, M = list(map(int, input().split()))

    AB = [[int(x) for x in input().split()] for _ in range(M)]

    Q = int(eval(input()))

    VDC = [[int(x) for x in input().split()] for _ in range(Q)]

    graph = [[] for _ in range(N + 1)]

    for a, b in AB:

        graph[a].append(b)

        graph[b].append(a)

    color = [0] * (N + 1)

    visited = [[False] * 11 for _ in range(N + 1)]

    # 残り体力 x で訪れた

    # 10で訪れたら、9,8,7も埋める

    for v, d, c in VDC[::-1]:

        if visited[v][d]:

            continue

        if color[v] == 0:

            color[v] = c

        for i in range(d + 1):

            visited[v][i] = True

        q = [v]

        for t in range(d - 1, -1, -1):

            qq = []

            for v in q:

                for w in graph[v]:

                    if visited[w][t]:

                        continue

                    for i in range(t + 1):

                        visited[w][i] = True

                    if color[w] == 0:

                        color[w] = c

                    qq.append(w)

            q = qq

    print(("\n".join(map(str, color[1:]))))


problem_p03768()
