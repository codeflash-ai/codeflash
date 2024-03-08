def problem_p02758():
    mod = 998244353

    N = int(eval(input()))

    X = []

    for _ in range(N):

        x, d = list(map(int, input().split()))

        X.append((x, x + d))

    X = sorted(X, key=lambda x: x[0])

    L = [(0, -(10**9 + 1), 10**9 + 1)]

    P = [-1] * (N + 1)

    for i, (x, y) in enumerate(X):

        while L[-1][2] <= y:

            if L[-1][2] > x and P[i + 1] < 0:

                P[i + 1] = L[-1][0]

            L.pop()

        if P[i + 1] < 0:

            P[i + 1] = L[-1][0]

        L.append((i + 1, x, y))

    C = [[] for _ in range(N + 1)]

    for i, p in enumerate(P):

        if p >= 0:

            C[p].append(i)

    Y = [1] * (N + 1)

    for i in range(N + 1)[::-1]:

        s = 1

        for j in C[i]:

            s = s * (Y[j] + 1) % mod

        Y[i] = s

    print((Y[0]))


problem_p02758()
