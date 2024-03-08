def problem_p03330():
    from collections import defaultdict

    N, C = list(map(int, input().split()))

    D = [tuple(map(int, input().split())) for _ in range(C)]

    d = [defaultdict(int) for i in range(3)]

    for i in range(N):

        c = [int(x) for x in input().split()]

        for j in range(N):

            for k in range(C):

                d[(i + j) % 3][k] += D[c[j] - 1][k]

    # print(d)

    ans = 10**9

    for i in range(C):

        for j in range(C):

            for k in range(C):

                if i != j and j != k and k != i:

                    ans = min(ans, d[0][i] + d[1][j] + d[2][k])

    print(ans)


problem_p03330()
