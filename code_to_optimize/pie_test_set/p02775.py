def problem_p02775():
    def main():

        n = list(eval(input()))

        n = [int(i) for i in n]

        l = len(n)

        dp = [[0] * 2 for _ in range(l + 1)]

        if n[0] >= 5:

            dp[0][1] = 1

        for i in range(l):

            dp[i + 1][0] = min(dp[i][0] + n[i], dp[i][1] + 10 - n[i])

            dp[i + 1][1] = min(dp[i][0] + n[i] + 1, dp[i][1] + 9 - n[i])

        # print(dp)

        print((dp[l][0]))

    if __name__ == "__main__":

        main()


problem_p02775()
