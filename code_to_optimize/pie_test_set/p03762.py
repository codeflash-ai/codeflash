def problem_p03762():
    MOD = 10**9 + 7

    n, m = list(map(int, input().split()))

    x = list(map(int, input().split()))

    y = list(map(int, input().split()))

    # 前計算

    for i in range(n - 1):

        x[i] = x[i + 1] - x[i]

    for i in range(m - 1):

        y[i] = y[i + 1] - y[i]

    N = [n - 1]

    for i in range(n - 3, 0, -2):

        N.append(N[-1] + i)

        # N.append((N[i] + (n - 1 - 2 * (i + 1))) % MOD)

    M = [m - 1]

    for i in range(m - 3, 0, -2):

        M.append(M[-1] + i)

        # M.append((M[i] + (m - 1- 2 * (i + 1))) % MOD)

    if N[0] % 2 == 0:

        N += N[::-1]

    elif N[0] != 1:

        N += N[-2::-1]

    if M[0] % 2 == 0:

        M += M[::-1]

    elif M[0] != 1:

        M += M[-2::-1]

    # print (x)

    # print (y)

    # print (N)

    # print (M)

    if len(x) - 1 > len(N):

        while True:

            pass

    if len(y) - 1 > len(M):

        while True:

            pass

    X_sum = 0

    for i in range(n - 1):

        X_sum += x[i] * N[i]

        X_sum %= MOD

    Y_sum = 0

    for i in range(m - 1):

        Y_sum += y[i] * M[i]

        Y_sum %= MOD

    print(((X_sum * Y_sum) % MOD))


problem_p03762()
