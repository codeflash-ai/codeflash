def problem_p00522():
    INF = 10**20

    m, n = list(map(int, input().split()))

    manju_lst = [int(eval(input())) for i in range(m)]

    manju_lst.sort(reverse=True)

    acc = 0

    cum_sum = [0]

    for manju in manju_lst:

        acc += manju

        cum_sum.append(acc)

    clst = []

    elst = []

    for i in range(n):

        c, e = list(map(int, input().split()))

        clst.append(c)

        elst.append(e)

    dp = [[INF] * (m + 1) for _ in range(n)]

    for i in range(n):

        dp[i][0] = 0

    # dp[x][y]...x種類目までの箱でy個以下売る時の最小コスト

    # dp[x][y] = min(dp[x - 1][y], dp[x - 1][y - cx] + ex) if (y - cx >= 0) else min(dp[x - 1][y], dp[x - 1][y + 1])

    for x in range(n):

        cx = clst[x]

        ex = elst[x]

        for y in range(m, 0, -1):

            if y >= cx:

                dp[x][y] = min(dp[x - 1][y], dp[x - 1][y - cx] + ex)

            else:

                if y + 1 <= m:

                    dp[x][y] = min(dp[x - 1][y], dp[x][y + 1])

                else:

                    dp[x][y] = min(dp[x - 1][y], ex)

    print((max([cum_sum[x] - dp[n - 1][x] for x in range(m + 1)])))


problem_p00522()
