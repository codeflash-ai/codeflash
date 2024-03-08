def problem_p03732():
    from collections import defaultdict as dd

    from itertools import accumulate as ac

    from itertools import product as pr

    N, W = list(map(int, input().split()))

    d = dd(list)

    for _ in range(N):

        w, v = list(map(int, input().split()))

        d[w].extend([v])

    d = dict(d)

    for i in list(d.keys()):

        d[i].sort(reverse=True)

        d[i] = [0] + list(ac(d[i]))

    l = []

    key = list(d.keys())

    for val in pr(*list(map(enumerate, list(d.values())))):

        weight = sum(map(lambda x, y: x * y[0], key, val))

        value = sum([x[1] for x in val])

        if weight <= W:

            l.extend([value])

    print((max(l)))


problem_p03732()
