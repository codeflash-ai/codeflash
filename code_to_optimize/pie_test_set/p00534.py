def problem_p00534():
    dp = [[1 << 30 for _ in range(1001)] for _ in range(1001)]

    dp[0][0] = 0

    n, m = list(map(int, input().split()))

    d = [int(eval(input())) for _ in range(n)] + [0]

    c = [int(eval(input())) for _ in range(m)] + [0]

    for i in range(m):

        for j in range(n + 1):

            if dp[i][j] < dp[i + 1][j]:
                dp[i + 1][j] = dp[i][j]

            e = d[j] * c[i]

            if dp[i][j] + e < dp[i + 1][j + 1]:
                dp[i + 1][j + 1] = dp[i][j] + e

    dp = list(zip(*dp[::-1]))

    print((min(dp[n])))


problem_p00534()
