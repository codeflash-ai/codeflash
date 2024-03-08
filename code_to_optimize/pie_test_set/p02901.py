def problem_p02901():
    INF = 10**10

    N, M = list(map(int, input().split()))

    costs = []

    keys = []

    for _ in range(M):

        A, B = list(map(int, input().split()))

        cs = list(map(int, input().split()))

        costs.append(A)

        # 鍵を2進数化する

        key = 0

        for c in cs:

            key |= 1 << (c - 1)

        keys.append(key)

    # [1]の個数で分類する

    maskss = [[] for _ in range(N + 1)]

    num1s = [0] * (2**N)

    for S in range(2**N):

        num = bin(S).count("1")

        maskss[num].append(S)

        num1s[S] = num

    dpAll = [INF] * (2**N)

    for S in range(2**N):

        for cost, key in zip(costs, keys):

            if S & key == S:

                if cost < dpAll[S]:

                    dpAll[S] = cost

    dp = [INF] * (2**N)

    for S in range(2**N):

        cost = dpAll[S]

        num1 = num1s[S]

        for k in range(1, (num1 + 1) // 2 + 1):

            for mask in maskss[k]:

                if mask & S == mask:

                    m2 = S ^ mask

                    c2 = dp[mask] + dp[m2]

                    if c2 < cost:

                        cost = c2

        dp[S] = cost

    if dp[2**N - 1] == INF:

        print((-1))

    else:

        print((dp[2**N - 1]))


problem_p02901()
