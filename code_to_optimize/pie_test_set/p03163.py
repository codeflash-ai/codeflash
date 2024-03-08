def problem_p03163():
    N, W = list(map(int, input().split()))

    w, v = [], []

    for i in range(N):

        a, b = list(map(int, input().split()))

        w.append(a)

        v.append(b)

    dp = [[0] * (W + 1) for i in range(N + 1)]

    for i in range(N):

        for j in range(W + 1):

            if j - w[i] >= 0:

                dp[i + 1][j] = dp[i][j - w[i]] + v[i]

            dp[i + 1][j] = max(dp[i + 1][j], dp[i][j])

    print((dp[N][W]))


problem_p03163()
