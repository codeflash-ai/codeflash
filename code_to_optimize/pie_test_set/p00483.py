def problem_p00483():
    def solve():

        m, n = list(map(int, input().split()))

        k = int(eval(input()))

        mp = [eval(input()) for i in range(m)]

        dp = [[[0] * 3 for j in range(n + 1)] for i in range(m + 1)]

        for i in range(1, m + 1):

            for j in range(1, n + 1):

                for p in range(3):

                    dp[i][j][p] = dp[i][j - 1][p] + dp[i - 1][j][p] - dp[i - 1][j - 1][p]

                if mp[i - 1][j - 1] == "J":

                    dp[i][j][0] += 1

                elif mp[i - 1][j - 1] == "O":

                    dp[i][j][1] += 1

                else:

                    dp[i][j][2] += 1

        for i in range(k):

            x1, y1, x2, y2 = list(map(int, input().split()))

            print(
                (
                    dp[x2][y2][0] - dp[x1 - 1][y2][0] - dp[x2][y1 - 1][0] + dp[x1 - 1][y1 - 1][0],
                    dp[x2][y2][1] - dp[x1 - 1][y2][1] - dp[x2][y1 - 1][1] + dp[x1 - 1][y1 - 1][1],
                    dp[x2][y2][2] - dp[x1 - 1][y2][2] - dp[x2][y1 - 1][2] + dp[x1 - 1][y1 - 1][2],
                )
            )

    solve()


problem_p00483()
