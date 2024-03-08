def problem_p03430():
    def main():

        s = eval(input())

        n = len(s)

        k = int(eval(input()))

        # dp[使った回数][左端からの距離][左端]

        dp = [[[1] * (n - i) for i in range(n)] for _ in range(k + 1)]

        for i in range(n - 1):

            if s[i] == s[i + 1]:

                dp[0][1][i] = 2

        if k > 0:

            for cnt in range(1, k + 1):

                for i in range(n - 1):

                    dp[cnt][1][i] = 2

        for cnt in range(k):

            for d in range(2, n):

                for left in range(n - d):

                    right = left + d

                    plus = (s[left] == s[right]) * 2

                    dp[cnt][d][left] = max(
                        dp[cnt][d][left],
                        dp[cnt][d - 1][left],
                        dp[cnt][d - 1][left + 1],
                        dp[cnt][d - 2][left + 1] + plus,
                    )

                    dp[cnt + 1][d][left] = max(
                        dp[cnt + 1][d][left], dp[cnt][d][left], dp[cnt][d - 2][left + 1] + 2
                    )

        for d in range(2, n):

            for left in range(n - d):

                right = left + d

                plus = (s[left] == s[right]) * 2

                dp[k][d][left] = max(
                    dp[k][d][left],
                    dp[k][d - 1][left],
                    dp[k][d - 1][left + 1],
                    dp[k][d - 2][left + 1] + plus,
                )

        print((dp[-1][-1][0]))

    main()


problem_p03430()
