def problem_p02662():
    import sys

    sys.setrecursionlimit(10000000)

    MOD = 998244353

    INF = 10**15

    def main():

        N, S = list(map(int, input().split()))

        A = list(map(int, input().split()))

        dp = [0] * (1 + S)

        dp[0] = pow(2, N, MOD)

        inv2 = pow(2, MOD - 2, MOD)

        for a in A:

            for s in range(S, a - 1, -1):

                dp[s] = (dp[s] + dp[s - a] * inv2) % MOD

        print((dp[S]))

    if __name__ == "__main__":

        main()


problem_p02662()
