def problem_p03787():
    import sys

    input = sys.stdin.readline

    N, M = list(map(int, input().split()))

    UV = [[int(x) for x in row.split()] for row in sys.stdin.readlines()]

    graph = [[] for _ in range(N + 1)]

    for u, v in UV:

        graph[u].append(v)

        graph[v].append(u)

    color = [None] * (N + 1)

    def calc_comp_data(v):

        c = 0

        q = [v]

        color[v] = 0

        is_bipartite = True

        size = 1

        while q:

            qq = []

            c ^= 1

            for x in q:

                for y in graph[x]:

                    if color[y] is None:

                        color[y] = c

                        qq.append(y)

                        size += 1

                    elif color[y] == c:

                        continue

                    else:

                        is_bipartite = False

            q = qq

        return size, is_bipartite

    size = []

    is_bipartite = []

    for v in range(1, N + 1):

        if color[v] is not None:

            continue

        x, y = calc_comp_data(v)

        size.append(x)

        is_bipartite.append(y)

    size, is_bipartite

    n_point = sum(1 if s == 1 else 0 for s in size)

    n_component = len(size)

    n_bipartitle = sum(1 if s >= 2 and bl else 0 for s, bl in zip(size, is_bipartite))

    answer = 0

    for s, bl in zip(size, is_bipartite):

        if s == 1:

            # 第1成分が1点のとき

            answer += N

        elif not bl:

            # 第1成分が2点以上、非二部グラフ

            # 成分の個数が足される

            answer += (n_component - n_point) + s * n_point

        else:

            # 第1成分が2点以上の二部グラフ

            # 相手が2点以上の二部グラフなら、2つできる

            answer += (n_component - n_point) + n_bipartitle + s * n_point

    print(answer)


problem_p03787()
