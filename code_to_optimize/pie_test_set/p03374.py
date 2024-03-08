def problem_p03374():
    import sys

    N, C = list(map(int, input().split()))

    table = []

    for i in range(N):

        x, v = list(map(int, input().split()))

        table.append([x, v])

    ans = 0

    if N == 1:

        ans = max(0, table[0][1] - table[0][0], table[0][1] + table[0][0] - C)

        print(ans)

        sys.exit()

    V = [0] * N

    clo = [0] * N

    V[0] = table[0][1]

    clo[0] = V[0] - table[0][0]

    for i in range(1, N):

        V[i] = V[i - 1] + table[i][1]

        clo[i] = max(clo[i - 1], V[i] - table[i][0])

        ans = max(clo[i], ans)

    rclo = [0] * N

    rclo[N - 1] = V[N - 1] - V[N - 2] - C + table[N - 1][0]

    for i in range(N - 1, 1, -1):

        rclo[i - 1] = max(rclo[i], V[N - 1] - V[i - 2] - C + table[i - 1][0])

        ans = max(clo[i], ans)

    rclo[0] = max(rclo[1], V[N - 1] - C + table[0][0])

    ans = max(rclo[0], ans)

    for i in range(N - 1):

        t = V[i] + rclo[i + 1] - 2 * table[i][0]

        ans = max(t, ans)

    for i in range(N - 1, 0, -1):

        t = V[N - 1] - V[i - 1] + clo[i - 1] - 2 * C + 2 * table[i][0]

        ans = max(t, ans)

    print(ans)


problem_p03374()
