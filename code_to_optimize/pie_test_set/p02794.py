def problem_p02794():
    N = int(eval(input()))

    X = [[] for i in range(N)]

    for i in range(N - 1):

        x, y = list(map(int, input().split()))

        X[x - 1].append(y - 1)

        X[y - 1].append(x - 1)

    P = [-1] * N

    DE = [0] * N

    Q = [0]

    while Q:

        i = Q.pop()

        for a in X[i][::-1]:

            if a != P[i]:

                P[a] = i

                DE[a] = DE[i] + 1

                X[a].remove(i)

                Q.append(a)

    def lp(u, v):

        t = 0

        while u != v:

            if DE[u] > DE[v]:

                t += 1 << u - 1

                u = P[u]

            elif DE[u] < DE[v]:

                t += 1 << v - 1

                v = P[v]

            else:

                t += 1 << u - 1

                t += 1 << v - 1

                u = P[u]

                v = P[v]

        return t

    Y = []

    M = int(eval(input()))

    for _ in range(M):

        a, b = list(map(int, input().split()))

        a, b = a - 1, b - 1

        Y.append(lp(a, b))

    D = {1 << i: i for i in range(50)}

    Z = [0] * (1 << M)

    ans = 0

    CC = [0] * N

    BC = [0] * (1 << 17)

    for m in range(1, 1 << 17):

        a = m & (-m)

        BC[m] = BC[m ^ a] + 1

    for m in range(1 << M):

        a = m & (-m)

        if a == m:

            if a == 0:

                Z[m] = 0

            else:

                Z[m] = Y[D[a]]

        else:

            Z[m] = Z[m ^ a] | Y[D[a]]

        aa = Z[m]

        bc = BC[aa % (1 << 17)]

        aa >>= 17

        bc += BC[aa % (1 << 17)]

        aa >>= 17

        bc += BC[aa]

        CC[N - 1 - bc] += 1 if BC[m % 1024] + BC[m >> 10] & 1 == 0 else -1

    print((sum([2**i * CC[i] for i in range(N)])))


problem_p02794()
