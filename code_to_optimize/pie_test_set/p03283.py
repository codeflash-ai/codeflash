def problem_p03283():
    n, m, q = list(map(int, input().split()))

    lr = [[0 for i in range(n + 1)] for j in range(n + 1)]

    # lr = [[int(i) for i in input().split()]for j in range(m)]

    st = [0] * (n + 1)

    ed = [0] * (n + 1)

    for i in range(m):

        l, r = list(map(int, input().split()))

        lr[l][r] += 1

    pq = [[int(i) for i in input().split()] for j in range(q)]

    dp = [[0 for i in range(n + 1)] for j in range(n + 1)]

    # print(lr)

    for i in range(1, n + 1):

        # print(i)

        for j in range(i, n + 1):

            # print(i,j)

            dp[i][j] = dp[i][j - 1] + lr[i][j]

    for i in range(q):

        st = pq[i][0]

        ed = pq[i][1]

        tmp = 0

        for j in range(ed - st + 1):

            tmp += dp[st + j][ed]

        print(tmp)

    # print(dp)


problem_p03283()
