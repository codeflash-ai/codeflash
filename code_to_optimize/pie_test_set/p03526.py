def problem_p03526():
    N = int(eval(input()))

    """
    
    dp[i][j] -> i番目の人間までみる。j人が積むときの最小値。
    
    
    
    """

    HP = [list(map(int, input().split())) for _ in range(N)]

    HP.sort(key=lambda x: x[0] + x[1])

    dp = [float("INF")] * (N + 1)

    dp[0] = 0

    for i in range(1, N + 1):

        for j in range(N, 0, -1):

            if dp[j - 1] <= HP[i - 1][0]:

                dp[j] = min(dp[j], dp[j - 1] + HP[i - 1][1])

    for j in range(N, -1, -1):

        if dp[j] != float("INF"):

            print(j)

            break


problem_p03526()
