def problem_p02314():
    n, m = list(map(int, input().split()))

    c = list(map(int, input().split()))

    INF = 10**10

    dp = [INF] * (n + 1)

    dp[0] = 0

    for i in range(n + 1):

        for j in range(len(c)):

            if i - c[j] >= 0:

                dp[i] = min(dp[i], dp[i - c[j]] + 1)

    print((dp[n]))


problem_p02314()
