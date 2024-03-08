def problem_p03160():
    n, h = int(eval(input())), list(map(int, input().split()))

    dp = [0, abs(h[1] - h[0])] + [0] * (n - 2)

    for i in range(2, n):
        dp[i] = min(dp[i - 1] + abs(h[i] - h[i - 1]), dp[i - 2] + abs(h[i] - h[i - 2]))

    print((dp[n - 1]))


problem_p03160()
