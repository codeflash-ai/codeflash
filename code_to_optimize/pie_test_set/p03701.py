def problem_p03701():
    N = int(eval(input()))

    S = [int(eval(input())) for i in range(N)]

    dp = {}

    dp[0] = True

    for s in S:

        keys = list(dp.keys())

        for k in keys:

            dp[k + s] = True

    res = 0

    for k in list(dp.keys()):

        if k % 10 == 0:

            continue

        res = max(res, k)

    print(res)


problem_p03701()
