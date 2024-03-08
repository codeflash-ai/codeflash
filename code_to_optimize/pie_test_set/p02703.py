def problem_p02703():
    from heapq import *

    (n, m, s), *t = [list(map(int, t.split())) for t in open(0)]

    (*e,) = eval("[]," * 8**6)

    for u, v, a, b in t[:m]:

        for i in range(2501 - a):

            e[(i + a) * 51 + u] += ((b, i * 51 + v),)

            e[(i + a) * 51 + v] += ((b, i * 51 + u),)

    for i, (c, d) in enumerate(t[m:], 1):

        for j in range(2501 - c):

            e[j * 51 + i] += ((d, (j + c) * 51 + i),)

    d = [10**18] * 8**6

    f = [1] * 8**6

    q = [(0, min(2500, s) * 51 + 1)]

    while q:

        c, v = heappop(q)

        if f[v] < 1:
            continue

        d[v], f[v] = c, 0

        for p, w in e[v]:

            if f[w]:
                heappush(q, (c + p, w))

    for i in range(2, n + 1):
        print((min(d[i::51])))


problem_p02703()
