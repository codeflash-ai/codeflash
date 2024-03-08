def problem_p02750():
    from bisect import bisect_right

    import sys

    sys.setrecursionlimit(10**7)

    input = sys.stdin.readline

    n, t = list(map(int, input().split()))

    shop = []

    shop_a0 = []

    for i in range(n):

        a, b = list(map(int, input().split()))

        if a != 0:

            shop.append([a, b, a / (b + 1)])

        else:

            shop_a0.append(b)

    shop = sorted(shop, key=lambda x: -x[2])

    shop_a0 = sorted(shop_a0)

    n = len(shop)

    INF = 10**10

    dp = [[INF for _ in range(30)] for __ in range(n + 1)]

    dp[0][0] = 0

    m = min(n + 1, 30)

    for i in range(n):

        for j in range(m):

            dp[i + 1][j] = min(
                dp[i][j], dp[i][j - 1] + 1 + shop[i][0] * (dp[i][j - 1] + 1) + shop[i][1]
            )

            dp[i + 1][j - 1] = min(dp[i + 1][j - 1], dp[i][j - 1])

    ss = []

    if shop_a0:

        ss.append(1 + shop_a0[0])

        for i in range(1, len(shop_a0)):

            ss.append(ss[-1] + 1 + shop_a0[i])

    ans = 0

    for i in range(m):

        if t - dp[n][i] >= 0:

            ans = max(ans, bisect_right(ss, t - dp[n][i]) + i)

    print(ans)


problem_p02750()
