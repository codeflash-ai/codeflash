def problem_p03112():
    import bisect

    from scipy.sparse.csgraph import floyd_warshall as wf

    INF = 10**20

    A, B, Q = list(map(int, input().split()))

    s = [-INF] + [int(eval(input())) for _ in range(A)] + [INF]

    t = [-INF] + [int(eval(input())) for _ in range(B)] + [INF]

    for i in range(Q):

        q = int(eval(input()))

        b, d = bisect.bisect_right(s, q), bisect.bisect_right(t, q)

        ans = INF

        for S in [s[b - 1], s[b]]:

            for T in [t[d - 1], t[d]]:

                d1, d2 = abs(S - q) + abs(T - S), abs(T - q) + abs(S - T)

                ans = min(ans, d1, d2)

        print(ans)


problem_p03112()
