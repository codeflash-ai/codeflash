def problem_p03603():
    import sys

    input = sys.stdin.readline

    sys.setrecursionlimit(10**7)

    n = int(eval(input()))

    P = list(map(int, input().split()))

    T = [[] for _ in range(n)]

    for i, p in enumerate(P, 1):

        T[p - 1].append(i)

    X = list(map(int, input().split()))

    D = [[-1] * n for _ in range(2)]

    def dfs(color, v):

        if D[color][v] != -1:

            return D[color][v]

        l = len(T[v])

        x = X[v]

        dp = [[float("inf")] * (x + 1) for _ in range(l + 1)]

        dp[0][0] = 0

        for i, nv in enumerate(T[v]):

            for j in range(x + 1):

                if j >= X[nv]:

                    dp[i + 1][j] = min(dp[i + 1][j], dp[i][j - X[nv]] + dfs(color, nv))

                if j >= dfs(color ^ 1, nv):

                    dp[i + 1][j] = min(dp[i + 1][j], dp[i][j - dfs(color ^ 1, nv)] + X[nv])

        res = min(dp[l])

        D[color][v] = res

        return res

    ans = dfs(0, 0)

    if ans == float("inf"):

        print("IMPOSSIBLE")

    else:

        print("POSSIBLE")


problem_p03603()
