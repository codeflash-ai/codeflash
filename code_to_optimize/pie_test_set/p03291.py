def problem_p03291(input_data):
    #!/usr/bin/env python3

    import sys

    MOD = 1000000007  # type: int

    def solve(S: str):

        N = len(S)

        # 0:文字列の総数 1:Aの数  2:ABの数 3:ABCの数

        ## dp[i][j] = i文字目まで見たときに状態jの数

        ## answer: dp[N][3]

        dp = [[0] * 4 for _ in range(N + 1)]

        dp[0][0] = 1

        for i in range(N):

            for j in (0, 1, 2, 3):

                if S[i] == "?":

                    dp[i + 1][j] += dp[i][j] * 3

                else:

                    dp[i + 1][j] += dp[i][j]

                dp[i + 1][j] %= MOD

            ## カウント進める場合

            if S[i] == "A" or S[i] == "?":

                dp[i + 1][1] += dp[i][0]

                dp[i + 1][1] %= MOD

            if S[i] == "B" or S[i] == "?":

                dp[i + 1][2] += dp[i][1]

                dp[i + 1][2] %= MOD

            if S[i] == "C" or S[i] == "?":

                dp[i + 1][3] += dp[i][2]

                dp[i + 1][3] %= MOD

        return dp[N][3] % MOD

        return

    def main():

        def iterate_tokens():

            for line in sys.stdin:

                for word in line.split():

                    yield word

        tokens = iterate_tokens()

        S = next(tokens)  # type: str

        solve(S)

    if __name__ == "__main__":

        main()
