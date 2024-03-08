def problem_p00097():
    """

    now...今注目する値

    used...使った数字の数

    sum...それまでの合計



    dp[now][used][sum]...nowまででused個の数字を使って合計sumの場合の数



    dp[now][used][sum] = dp[now - 1][used - 1][sum - now] + dp[now - 1][used][sum] (used >= 1 and sum >= now)

                         dp[now - 1][used][sum]                                    (used == 0  or  sum < now)





    2次元化



    dp[used][sum]...used個の数字を使って合計sumの場合の数



    dp[used][sum] = dp[used - 1][sum - now] + dp[used][sum] (used >= 1 and sum >= now)



    ただし、usedの大きい順に更新する(更新がかぶるため)

    """

    dp = [[0 for _ in range(1001)] for _ in range(10)]

    dp[1][0] = 1

    dp[0][0] = 1

    for now in range(1, 101):

        for used in range(9, 0, -1):

            for s in range(now, 1001):

                dp[used][s] = dp[used - 1][s - now] + dp[used][s]

    while True:

        n, s = list(map(int, input().split()))

        if not n:

            break

        print((dp[n][s]))


problem_p00097()
