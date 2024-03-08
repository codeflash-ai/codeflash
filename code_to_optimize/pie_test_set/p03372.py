def problem_p03372():
    from itertools import accumulate as acc

    import sys

    N, C = list(map(int, input().split()))

    a = [list(map(int, input().split())) for i in range(N)]

    if N == 1:
        print((max(0, a[0][1] - a[0][0], a[0][1] - (C - a[0][0]))), sys.exit())

    b = list(acc([a[i][1] for i in range(N)]))

    ans = sum([a[i][1] for i in range(N)]) + max(C - a[0][0], a[N - 1][0])

    l = [[b[i] - a[i][0], 0] for i in range(N)]

    r = [[b[N - 1] - (C - a[0][0]), 0]] + [
        [b[N - 1] - b[i] - (C - a[i + 1][0]), 0] for i in range(N - 1)
    ]

    tmp = 0

    for i in range(N):

        if l[i][0] > tmp:
            tmp = l[i][0]

        l[i][1] = tmp

    tmp = 0

    for i in range(N - 1, -1, -1):

        if r[i][0] > tmp:
            tmp = r[i][0]

        r[i][1] = tmp

    p = [l[0][1], r[0][1], l[0][1] - a[0][0] + r[1][1]]

    q = [l[N - 1][1], r[N - 1][1], r[N - 1][0] - (C - a[N - 1][0]) + l[N - 2][1]]

    ans = max(p + q)

    for i in range(1, N - 1):

        p1 = l[i][1]

        p2 = r[i][1]

        p3 = l[i][1] - a[i][0] + r[i + 1][1]

        p4 = r[i][1] - (C - a[i][0]) + l[i - 1][1]

        ans = max(ans, p1, p2, p3, p4)

    print(ans)


problem_p03372()
