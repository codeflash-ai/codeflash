def problem_p03847():
    N = int(eval(input()))

    dp = [[0] * 3 for i in range(100)]

    MOD = int(1e9 + 7)

    maxBit = 70

    while not (N >> maxBit):

        maxBit -= 1

    dp[maxBit][0] = 1

    dp[maxBit][1] = 1

    for i in range(maxBit - 1, -1, -1):

        for j in range(3):

            bit = (N >> i) & 1

            for k in range(3):

                nj = 2 * j + bit - k

                if nj < 0:

                    continue

                nj = min((nj, 2))

                dp[i][nj] += dp[i + 1][j]

                dp[i][nj] %= MOD

    print((sum(dp[0]) % MOD))


problem_p03847()
