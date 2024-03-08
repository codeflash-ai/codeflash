def problem_p02863():
    n, t = list(map(int, input().split()))

    ab = [list(map(int, input().split())) for _ in range(n)]

    ab.sort()

    dp = [(6007) * [0] for _ in range(n + 1)]

    dp[0][0] = 0

    ans = 0

    for i in range(n):

        for j in range(6007):

            dp[i + 1][j] = max(dp[i + 1][j], dp[i][j])

            if j < t:
                dp[i + 1][j + ab[i][0]] = max(dp[i][j] + ab[i][1], dp[i][j + ab[i][0]])

            ans = max(ans, dp[i + 1][j])

    print(ans)


problem_p02863()
