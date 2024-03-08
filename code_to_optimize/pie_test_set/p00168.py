def problem_p00168():
    from math import ceil

    dp = [1] + [0 for i in range(30)]

    for i in range(1, 31):

        for step in [1, 2, 3]:

            if step <= i:

                dp[i] += dp[i - step]

    unit = 3650.0

    while 1:

        n = int(input())

        if n == 0:

            break

        print(int(ceil(dp[n] / unit)))


problem_p00168()
