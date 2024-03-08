def problem_p03064():
    n = int(eval(input()))

    a = [int(eval(input())) for i in range(n)]

    sum_a = sum(a)

    MOD = 998244353

    dp = [[0] * (sum_a + 1) for i in range(n + 1)]

    dp[0][0] = 1

    for i in range(n):

        for j in range(sum_a + 1):

            # 選ぶとき

            if j - a[i] >= 0:

                dp[i + 1][j] += dp[i][j - a[i]]

            # 選ばないとき

            dp[i + 1][j] += dp[i][j] * 2

            dp[i + 1][j] %= MOD

    dq = [[0] * (sum_a + 1) for i in range(n + 1)]

    dq[0][0] = 1

    for i in range(n):

        for j in range(sum_a + 1):

            # 選ぶとき

            if j - a[i] >= 0:

                dq[i + 1][j] += dq[i][j - a[i]]

            # 選ばないとき

            dq[i + 1][j] += dq[i][j]

            dq[i + 1][j] %= MOD

    ans = 0

    for j in range(sum_a + 1):

        if sum_a <= j * 2:

            ans += dp[-1][j] * 3

            ans %= MOD

    if sum_a % 2 == 0:

        ans -= dq[-1][sum_a // 2] * 3

    print(((3**n - ans) % MOD))


problem_p03064()
