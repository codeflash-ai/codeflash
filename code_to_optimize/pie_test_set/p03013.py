def problem_p03013():
    import numpy as np

    def resolve():

        MOD = 10**9 + 7

        n, m = list(map(int, input().split()))

        a = [int(eval(input())) for _ in range(m)]

        dp = np.array([1] * (n + 1))

        dp[a] = 0

        for i in range(2, n + 1):

            if dp[i] != 0:

                dp[i] = np.sum(dp[i - 2 : i]) % MOD

        print((dp[n]))

    resolve()


problem_p03013()
