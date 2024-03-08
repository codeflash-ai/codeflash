def problem_p00336():
    """

    dp[x][y] ... tのx文字目までで, y文字一致した場合の数

    dp[x][y] = dp[x - 1][y] + dp[x - 1][y - 1] * (t[x] == b[y])

    """

    MOD = 1000000007

    t = eval(input())

    b = eval(input())

    lent = len(t)

    lenb = len(b)

    dp = [[0] * (lenb + 1) for _ in range(lent + 1)]

    for i in range(lent + 1):

        dp[i][0] = 1

    for x in range(1, lent + 1):

        for y in range(1, lenb + 1):

            dp[x][y] = (dp[x - 1][y] + dp[x - 1][y - 1] * (t[x - 1] == b[y - 1])) % MOD

    print((dp[lent][lenb]))


problem_p00336()
