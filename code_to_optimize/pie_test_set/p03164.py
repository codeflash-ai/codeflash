def problem_p03164():
    def main():

        N, W = list(map(int, input().split()))

        w = [0] * N

        v = [0] * N

        for i in range(N):

            w[i], v[i] = list(map(int, input().split()))

        max_v = max(v)

        dp = [[float("inf")] * (max_v * N + 5) for _ in range(N + 5)]

        dp[0][0] = 0

        for i in range(N):

            for j in range(max_v * N):

                if j + v[i] <= max_v * N:

                    dp[i + 1][j + v[i]] = min(dp[i + 1][j + v[i]], dp[i][j] + w[i])

                dp[i + 1][j] = min(dp[i + 1][j], dp[i][j])

        ans = 0

        for j in range(max_v * N + 1):

            if dp[N][j] <= W:

                ans = j

        print(ans)

    if __name__ == "__main__":

        main()


problem_p03164()
