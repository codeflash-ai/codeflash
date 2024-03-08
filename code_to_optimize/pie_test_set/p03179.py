def problem_p03179():
    md = 10**9 + 7

    n = int(eval(input()))

    s = eval(input())

    dp = [1] * n

    for i in range(n - 1):

        if s[i] == "<":

            for dpi in range(1, n - i - 1):

                dp[dpi] = (dp[dpi] + dp[dpi - 1]) % md

            dp = dp[:-1]

        else:

            for dpi in range(n - i - 2, 0, -1):

                dp[dpi] = (dp[dpi] + dp[dpi + 1]) % md

            dp = dp[1:]

    print((dp[0]))


problem_p03179()
