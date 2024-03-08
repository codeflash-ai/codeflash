def problem_p02710():
    import sys

    input = lambda: sys.stdin.readline().rstrip()

    N = int(input())

    C = [int(a) - 1 for a in input().split()]

    X = [[] for i in range(N)]

    for i in range(N - 1):

        x, y = map(int, input().split())

        x, y = x - 1, y - 1

        X[x].append(y)

        X[y].append(x)

    def EulerTour(n, X, i0):

        def f(k):

            return k * (k + 1) // 2

        USED = [0] * n

        USEDORG = [0] * n

        USEDTMP = [0] * n

        P = [-1] * n

        Q = [~i0, i0]

        ct = -1

        ET = []

        ET1 = [0] * n

        ET2 = [0] * n

        DE = [0] * n

        ANS = [0] * n

        de = -1

        while Q:

            i = Q.pop()

            if i < 0:

                ET2[~i] = ct

                de -= 1

                USED[C[~i]] += 1 + USEDTMP[~i]

                USEDTMP[~i] = 0

                if ~i:

                    p = P[~i]

                    pc = C[p]

                    k = ET2[~i] - ET1[~i] + 1 - USED[pc] + USEDORG[~i]

                    ANS[pc] += f(k)

                    USEDTMP[p] += k

                continue

            if i >= 0:

                if i:

                    USEDORG[i] = USED[C[P[i]]]

                ET.append(i)

                ct += 1

                if ET1[i] == 0:
                    ET1[i] = ct

                de += 1

                DE[i] = de

            for a in X[i][::-1]:

                if a != P[i]:

                    P[a] = i

                    for k in range(len(X[a])):

                        if X[a][k] == i:

                            del X[a][k]

                            break

                    Q.append(~a)

                    Q.append(a)

        for i in range(n):

            ANS[i] = f(n) - f(n - USED[i]) - ANS[i]

        return ANS

    ANS = EulerTour(N, X, 0)

    print(*ANS, sep="\n")


problem_p02710()
