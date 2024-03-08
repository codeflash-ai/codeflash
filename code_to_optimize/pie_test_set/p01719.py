def problem_p01719():
    import sys

    def calc(x, n, p1, p2):

        dp = []

        for i in range(n):

            dp.append([0] * (x + 1))

        pocket = x

        for i in range(n):

            for j in range(x + 1):

                dp[i][j] = dp[i - 1][j]

                if j - p1[i] >= 0:

                    dp[i][j] = max(dp[i][j], dp[i][j - p1[i]] + p2[i])

                pocket = max(pocket, (x - j) + dp[i][j])

        return pocket

    def main():

        n, d, x = list(map(int, sys.stdin.readline().split()))

        pp = []

        for i in range(d):

            pp.append(list(map(int, sys.stdin.readline().split())))

        curr_x = x

        for i in range(d - 1):

            curr_x = calc(curr_x, n, pp[i], pp[i + 1])

        print(curr_x)

    if __name__ == "__main__":

        main()


problem_p01719()
