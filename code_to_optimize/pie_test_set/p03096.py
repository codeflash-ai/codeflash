def problem_p03096():
    n = int(eval(input()))

    c = [int(eval(input())) for _ in range(n)]

    mod = 10**9 + 7

    # 解説AC

    dp = [1] * n

    color = [-1] * max(c)

    for i, ci in enumerate(c):

        dp[i] = dp[i - 1]

        if 0 <= color[ci - 1] < i - 1:

            dp[i] += dp[color[ci - 1]]

        color[ci - 1] = i

    print((dp[-1] % mod))


problem_p03096()
