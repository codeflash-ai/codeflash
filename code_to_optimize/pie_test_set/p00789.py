def problem_p00789():
    # AOJ 1209: Square Coins

    # Python3 2018.7.19 bal4u

    N = 18

    tbl = [i**2 for i in range(0, N)]

    dp = [[0 for j in range(600)] for i in range(N)]

    dp[0][0] = 1

    for i in range(1, N):

        for n in range(300):

            dp[i][n] += dp[i - 1][n]

            for j in range(tbl[i], 300, tbl[i]):

                dp[i][n + j] += dp[i - 1][n]

    while True:

        n = int(eval(input()))

        if n == 0:
            break

        print((dp[N - 1][n]))


problem_p00789()
