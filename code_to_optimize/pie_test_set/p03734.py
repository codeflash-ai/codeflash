def problem_p03734():
    N, W = list(map(int, input().split()))

    ws = []

    vs = []

    for i in range(N):

        w, v = list(map(int, input().split()))

        ws.append(w)

        vs.append(v)

    ws_m = [w - ws[0] for w in ws]

    """
    
    i番目まで見て、n個選んでいて、重さがwのときのvの最大値
    
    ただし、wはN個選んだときにW-N*[w[0]]となっているので、そこをmaxとする
    
    dp[i][n][w] 
    
    """

    dp = [[[-1] * (3 * N + 1) for i in range(N + 1)] for j in range(N + 1)]

    dp[0][0][0] = 0

    for i, (w, v) in enumerate(zip(ws_m, vs)):

        for n in range(i + 1):

            for j in range(3 * N):

                if dp[i][n][j] == -1:
                    continue

                dp[i + 1][n + 1][j + w] = max(dp[i][n][j] + v, dp[i][n][j + w])

                dp[i + 1][n][j] = max(dp[i + 1][n][j], dp[i][n][j])

    ans = 0

    for i, a in enumerate(dp[N]):

        w = W - i * ws[0]

        if w >= 0:

            ans = max(ans, max(a[: w + 1]))

    print(ans)


problem_p03734()
