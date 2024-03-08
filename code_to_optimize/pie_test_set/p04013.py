def problem_p04013():
    from collections import Counter

    def main():

        num, avg = list(map(int, input().split()))

        data = list(map(int, input().split()))

        dp = [[[0 for i in range(6000)] for j in range(num + 1)] for k in range(num + 1)]

        dp[0][0][0] = 1

        for i in range(1, num + 1):

            now_card = data[i - 1]

            for j in range(num + 1):

                for k in range(6000):

                    # print(i, j, k)

                    dp[i][j][k] += dp[i - 1][j][k]

                    if k - now_card >= 0 and j - 1 >= 0:

                        dp[i][j][k] += dp[i - 1][j - 1][k - now_card]

            # print(dp[i][j])

        ans = 0

        for i in range(1, num + 1):

            ans += dp[num][i][i * avg]

        print(ans)

    if __name__ == "__main__":

        main()


problem_p04013()
