def problem_p03806():
    N, Ma, Mb = list(map(int, input().split()))

    arr = [list(map(int, input().split())) for _ in range(N)]

    dp = [[[5000] * 401 for _ in range(401)] for _ in range(41)]

    dp[0][0][0] = 0

    for i in range(N):

        a, b, c = arr[i]

        for j in range(10 * N + 1):

            for k in range(10 * N + 1):

                if dp[i][j][k] == 5000:

                    continue

                dp[i + 1][j][k] = min(dp[i + 1][j][k], dp[i][j][k])

                dp[i + 1][j + a][k + b] = min(dp[i + 1][j + a][k + b], dp[i][j][k] + c)

    ans = 5000

    for i in range(1, 401):

        if i * Ma > 400 or i * Mb > 400:

            break

        ans = min(ans, dp[N][i * Ma][i * Mb])

    if ans == 5000:

        print((-1))

    else:

        print(ans)


problem_p03806()
