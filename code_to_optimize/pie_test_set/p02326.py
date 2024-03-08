def problem_p02326():
    import sys

    def solve():

        file_input = sys.stdin

        H, W = list(map(int, file_input.readline().split()))

        dp = [[0] * (W + 1)]

        for line in file_input:

            dp.append([0] + list(map(int, line.split())))

        max_width = 0

        for i in range(1, H + 1):

            for j in range(1, W + 1):

                if dp[i][j] == 1:

                    dp[i][j] = 0

                else:

                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1

                    max_width = max(max_width, dp[i][j])

        print((max_width**2))

    solve()


problem_p02326()
