def problem_p00729():
    while 1:

        N, M = list(map(int, input().split()))

        if N == 0:
            break

        T = [[0] * 1261 for i in range(N)]

        for i in range(eval(input())):

            t, n, m, s = list(map(int, input().split()))

            for ti in range(t, 1261):

                T[n - 1][ti] = s * m

        for i in range(eval(input())):

            ts, te, m = list(map(int, input().split()))

            print(sum(1 if any(T[n][t] == m for n in range(N)) else 0 for t in range(ts, te)))


problem_p00729()
