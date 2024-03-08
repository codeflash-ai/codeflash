def problem_p02666():
    def productall(a):

        N = len(a)

        if N == 0:
            return [1]

        A = [1] * N + a[:]

        P = 10**9 + 7

        k = 100

        K = k * 8

        pa1 = (1 << k * 4 + 16) - ((1 << k * 4 + 16) % P)

        pa2 = (1 << k * 2 + 24) - ((1 << k * 2 + 24) % P)

        pa3 = (1 << k + 28) - ((1 << k + 28) % P)

        def modP(x):

            x -= ((x & m1) >> k * 4 + 16) * pa1

            x -= ((x & m2) >> k * 2 + 24) * pa2

            x -= ((x & m3) >> k + 28) * pa3

            return x

        for i in range(N)[::-1]:

            if i == N - 1 or i and 1 << i.bit_length() - 1 == i:

                if i == N - 1:

                    c = 2

                else:

                    c *= 2

                m1 = int(("1" * (k * 4 - 16) + "0" * (k * 4 + 16)) * c, 2)

                m2 = int(("1" * (k * 6 - 24) + "0" * (k * 2 + 24)) * c, 2)

                m3 = int(("1" * (k * 7 - 28) + "0" * (k + 28)) * c, 2)

            A[i] = modP(A[2 * i] * A[2 * i + 1])

        t = bin(A[1])[2:] + "_"

        return [int(t[-(i + 1) * K - 1 : -i * K - 1], 2) % P for i in range((len(t) + K - 2) // K)]

    def par(a):

        L = []

        while P[a] != a:

            L.append(a)

            a = P[a]

        for l in L:

            P[l] = a

        return a

    def unite(a, b):

        pa = par(a)

        pb = par(b)

        if pa == pb:
            return

        if LEN[pa] < LEN[pb]:

            a, b, pa, pb = b, a, pb, pa

        P[pb] = pa

        if LEN[pa] == LEN[pb]:
            LEN[pa] += 1

        CNT[pa] += CNT[pb]

    def cnt(a):

        return CNT[par(a)]

    N = int(eval(input()))

    P = [i for i in range(N)]

    LEN = [1] * N

    CNT = [1] * N

    FULL = [0] * N

    A = [int(a) - 1 for a in input().split()]

    for i, a in enumerate(A):

        if a < 0:
            continue

        if par(i) != par(a):

            unite(i, a)

        else:

            FULL[i] = 1

    for i in range(N):

        if FULL[i]:

            FULL[par(i)] = 1

    X = []

    Y = []

    for i in range(N):

        if par(i) == i:

            if FULL[i] == 0:

                X.append(CNT[i])

            else:

                Y.append(CNT[i])

    M = len(X)

    mod = 10**9 + 7

    ans = (sum(X) + sum(Y) - len(Y)) * pow(N - 1, M, mod) % mod

    L = productall([(a << 800) + 1 for a in X])

    if M:

        fa = 1

        ans = (ans + M * pow(N - 1, M - 1, mod)) % mod

        for i, l in enumerate(L):

            if i == 0:
                continue

            ans = (ans - l * fa * pow(N - 1, M - i, mod)) % mod

            fa = fa * i % mod

    print(ans)


problem_p02666()
