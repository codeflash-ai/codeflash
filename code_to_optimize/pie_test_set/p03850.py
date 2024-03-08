def problem_p03850():
    n = eval(input())

    code = input().split()

    # DP(N, <parentheses nested>)

    dp = [[0] * 3 for i in range(n)]

    dp[0] = [int(code[0]), -(10**18), -(10**18)]

    for i in range(1, n):

        op, a = code[2 * i - 1 : 2 * i + 1]

        v = int(a)

        if op is "+":

            dp[i][0] = max(dp[i - 1][0] + v, dp[i - 1][1] - v, dp[i - 1][2] + v)

            dp[i][1] = max(dp[i - 1][1] - v, dp[i - 1][2] + v)

            dp[i][2] = dp[i - 1][2] + v

        else:

            dp[i][0] = max(dp[i - 1][0] - v, dp[i - 1][1] + v, dp[i - 1][2] - v)

            dp[i][1] = max(dp[i - 1][0] - v, dp[i - 1][1] + v, dp[i - 1][2] - v)

            dp[i][2] = max(dp[i - 1][1] + v, dp[i - 1][2] - v)

    print(max(dp[-1]))


problem_p03850()
