def problem_p03082():
    N, X = list(map(int, input().split()))

    S = list(map(int, input().split()))

    S.sort()

    mod = 10**9 + 7

    dp = [[x] + [0] * N for x in range(X + 1)]

    for x in range(X + 1):

        for n in range(1, N + 1):

            dp[x][n] = (dp[x % S[n - 1]][n - 1] + (n - 1) * dp[x][n - 1]) % mod

    print((dp[X][N]))


problem_p03082()
