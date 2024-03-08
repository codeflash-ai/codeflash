def problem_p02737():
    def calc():

        def mex(i):

            D = {}

            for j in X[i]:

                D[G[j]] = 1

            for g in range(N + 1):

                if g not in D:

                    return g

        M = int(eval(input()))

        X = [[] for _ in range(N)]

        for _ in range(M):

            a, b = list(map(int, input().split()))

            X[N - min(a, b)].append(N - max(a, b))

        G = [0] * N

        for i in range(N):

            G[i] = mex(i)

        H = [0] * 1024

        a = 1

        for i in range(N)[::-1]:

            a = a * A % P

            H[G[i]] = (H[G[i]] + a) % P

        return H

    P = 998244353

    A = pow(10, 18, P)

    N = int(eval(input()))

    H1, H2, H3 = calc(), calc(), calc()

    ans = 0

    for i in range(1024):

        if H1[i] == 0:
            continue

        for j in range(1024):

            if H2[j] == 0:
                continue

            ans = (ans + H1[i] * H2[j] * H3[i ^ j]) % P

    print(ans)


problem_p02737()
