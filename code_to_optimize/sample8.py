def main():
    n = int(input())
    a, b, c = [0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    dp = [[0] * 3 for _ in range(n + 1)]

    for i in range(1, n + 1):
        a[i], b[i], c[i] = map(int, input().split())

    dp[1][0], dp[1][1], dp[1][2] = a[1], b[1], c[1]

    for i in range(2, n + 1):
        dp[i][0] = max(dp[i - 1][1], dp[i - 1][2]) + a[i]
        dp[i][1] = max(dp[i - 1][0], dp[i - 1][2]) + b[i]
        dp[i][2] = max(dp[i - 1][0], dp[i - 1][1]) + c[i]

    print(max(dp[n][0], dp[n][1], dp[n][2]))

if __name__ == "__main__":
    main()
