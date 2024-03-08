def problem_p01420():
    #!/usr/bin/env python

    from collections import deque

    import itertools as it

    import sys

    sys.setrecursionlimit(1000000)

    N, M, L = list(map(int, input().split()))

    def fact(n):

        if n <= 1:

            return 1

        return fact(n - 1) * n

    MC = []

    for i in range(M + 1):

        MC.append(fact(M) / fact(i) / fact(M - i))

    p_lst = []

    P_val = []

    P_sum = []

    for i in range(N):

        P, T, V = list(map(int, input().split()))

        p_lst.append((T, V))

        PP = []

        for k in range(M + 1):

            PP.append(float(MC[k] * (P**k) * ((100 - P) ** (M - k))) / 100**M)

        P_val.append(list(PP))

        for k in range(M, 0, -1):

            PP[k - 1] += PP[k]

        P_sum.append(PP)

    def comp(p1, p2, k1, k2):

        T1, V1 = p1

        T2, V2 = p2

        return L * (V2 - V1) < V1 * V2 * (k2 * T2 - k1 * T1)

    for i in range(N):

        ans = 0

        index_lst = [0 for j in range(N)]

        for k1 in range(M + 1):

            ret = P_val[i][k1]

            for j in range(N):

                if i == j:

                    continue

                flag = True

                while True:

                    k2 = index_lst[j]

                    if k2 > M:

                        ret *= 0

                        break

                    if comp(p_lst[i], p_lst[j], k1, k2):

                        ret *= P_sum[j][k2]

                        break

                    index_lst[j] += 1

            ans += ret

        print("%.10f" % ans)


problem_p01420()
