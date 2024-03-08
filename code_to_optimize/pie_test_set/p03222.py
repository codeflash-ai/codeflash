def problem_p03222():
    from itertools import product

    H, W, K = list(map(int, input().split()))

    mod = 10**9 + 7

    # dp[i][j]:= 上からi番目, 左からj番目にいる通り数

    dp = [[0] * W for i in range(H + 1)]

    dp[0][0] = 1

    def calc(x):

        if x < 0:

            return 1

        ret = 0

        for p in product(["0", "1"], repeat=x):

            p = "".join(p)

            if "11" not in p:

                ret += 1

        return ret

    for i in range(1, H + 1):

        for j in range(W):

            tmp = 0

            if j > 0:  # 左から

                tmp += dp[i - 1][j - 1] * calc(j - 2) * calc(W - j - 2) % mod

            # 上から

            tmp += dp[i - 1][j] * calc(j - 1) * calc(W - j - 2) % mod

            if j + 1 < W:  # 右から

                tmp += dp[i - 1][j + 1] * calc(j - 1) * calc(W - j - 3) % mod

            dp[i][j] = tmp

    print((dp[H][K - 1] % mod))


problem_p03222()
