def problem_p03003():
    n, m = list(map(int, input().split()))

    s = list(map(int, input().split()))

    t = list(map(int, input().split()))

    MOD = 10**9 + 7

    dp = [[0] * (m + 1) for i in range(n + 1)]

    for si in range(n + 1):

        dp[si][0] = 1

    for ti in range(m + 1):

        dp[0][ti] = 1

    for si in range(n):

        for ti in range(m):

            if s[si] == t[ti]:

                dp[si + 1][ti + 1] = dp[si][ti + 1] + dp[si + 1][ti]

                dp[si + 1][ti + 1] %= MOD

            else:

                dp[si + 1][ti + 1] = dp[si][ti + 1] + dp[si + 1][ti] - dp[si][ti]

                dp[si + 1][ti + 1] %= MOD

    print((dp[-1][-1] % MOD))


problem_p03003()
