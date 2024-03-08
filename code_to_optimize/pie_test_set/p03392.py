def problem_p03392():
    def solve(s):

        if len(s) == 2:

            return 1 if s[0] == s[1] else 2

        elif len(s) == 3:

            if s[0] == s[1] == s[2]:

                return 1

            elif s[0] == s[1] or s[1] == s[2]:

                return 6

            elif s[0] == s[2]:

                return 7

            else:

                return 3

        if all(a == b for a, b in zip(s, s[1:])):

            return 1

        dp = [[[0] * 3 for _ in range(3)] for _ in range(2)]

        dp[0][0][0] = 1

        dp[0][1][1] = 1

        dp[0][2][2] = 1

        MOD = 998244353

        for _ in range(len(s) - 1):

            ndp = [[[0] * 3 for _ in range(3)] for _ in range(2)]

            ndp[0][0][0] = (dp[0][1][0] + dp[0][2][0]) % MOD

            ndp[0][0][1] = (dp[0][1][1] + dp[0][2][1]) % MOD

            ndp[0][0][2] = (dp[0][1][2] + dp[0][2][2]) % MOD

            ndp[0][1][0] = (dp[0][0][2] + dp[0][2][2]) % MOD

            ndp[0][1][1] = (dp[0][0][0] + dp[0][2][0]) % MOD

            ndp[0][1][2] = (dp[0][0][1] + dp[0][2][1]) % MOD

            ndp[0][2][0] = (dp[0][0][1] + dp[0][1][1]) % MOD

            ndp[0][2][1] = (dp[0][0][2] + dp[0][1][2]) % MOD

            ndp[0][2][2] = (dp[0][0][0] + dp[0][1][0]) % MOD

            ndp[1][0][0] = (dp[0][0][0] + dp[1][0][0] + dp[1][1][0] + dp[1][2][0]) % MOD

            ndp[1][0][1] = (dp[0][0][1] + dp[1][0][1] + dp[1][1][1] + dp[1][2][1]) % MOD

            ndp[1][0][2] = (dp[0][0][2] + dp[1][0][2] + dp[1][1][2] + dp[1][2][2]) % MOD

            ndp[1][1][0] = (dp[0][1][2] + dp[1][0][2] + dp[1][1][2] + dp[1][2][2]) % MOD

            ndp[1][1][1] = (dp[0][1][0] + dp[1][0][0] + dp[1][1][0] + dp[1][2][0]) % MOD

            ndp[1][1][2] = (dp[0][1][1] + dp[1][0][1] + dp[1][1][1] + dp[1][2][1]) % MOD

            ndp[1][2][0] = (dp[0][2][1] + dp[1][0][1] + dp[1][1][1] + dp[1][2][1]) % MOD

            ndp[1][2][1] = (dp[0][2][2] + dp[1][0][2] + dp[1][1][2] + dp[1][2][2]) % MOD

            ndp[1][2][2] = (dp[0][2][0] + dp[1][0][0] + dp[1][1][0] + dp[1][2][0]) % MOD

            dp = ndp

        return (
            sum(dp[1][x][sum(map(ord, s)) % 3] for x in range(3))
            + all(a != b for a, b in zip(s, s[1:]))
        ) % MOD

    s = eval(input())

    print((solve(s)))


problem_p03392()
