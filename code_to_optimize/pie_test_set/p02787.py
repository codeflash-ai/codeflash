def problem_p02787():
    h, n = list(map(int, input().split()))

    magic = []

    INF = 10**8

    dp = [
        [INF for j in range(h + 1)] for i in range(n + 1)
    ]  # 体力がjのモンスターをi番目の魔法までを使って倒せる消費魔力の最小値

    for i in range(n):

        a, b = list(map(int, input().split()))

        magic.append([a, b])

    for i in range(n):

        for j in range(h):

            if j + 1 > magic[i][0]:

                dp[i + 1][j + 1] = min(
                    dp[i][j + 1],
                    dp[i][j + 1 - magic[i][0]] + magic[i][1],
                    dp[i + 1][j + 1 - magic[i][0]] + magic[i][1],
                )

            else:

                dp[i + 1][j + 1] = min(dp[i][j + 1], magic[i][1])

    print((dp[n][h]))


problem_p02787()
