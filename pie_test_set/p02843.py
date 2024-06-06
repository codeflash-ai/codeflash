def problem_p02843(input_data):
    X = int(eval(input_data))

    dp = [0] * (X + 105)

    dp[0] = 1

    for x in range(X):

        if dp[x] == 1:

            for price in range(100, 106):

                dp[x + price] = 1

    return dp[X]
