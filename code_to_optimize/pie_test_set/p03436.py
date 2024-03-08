def problem_p03436():
    # -*- coding: utf-8 -*-

    from queue import Queue

    h, w = list(map(int, input().split()))

    d = []

    for _ in range(h):

        d.append(eval(input()))

    cost = [[0] * w for _ in range(h)]

    q = Queue()

    q.put((0, 0))

    while not q.empty():

        hi, wi = q.get()

        if hi == h - 1 and wi == w - 1:

            break

        for dh, dw in [[0, 1], [1, 0], [0, -1], [-1, 0]]:

            if h > dh + hi >= 0 and w > dw + wi >= 0 and d[dh + hi][dw + wi] == ".":

                if cost[dh + hi][dw + wi] == 0 or cost[dh + hi][dw + wi] > cost[hi][wi] + 1:

                    cost[dh + hi][dw + wi] = cost[hi][wi] + 1

                    q.put((dh + hi, dw + wi))

    if cost[h - 1][w - 1] == 0:

        print((-1))

        exit()

    res = 0

    for hi in range(h):

        for wi in range(w):

            if (hi == 0 and wi == 0) or (hi == h - 1 and wi == -1):

                continue

            if d[hi][wi] == ".":

                res += 1

    print((res - cost[h - 1][w - 1]))


problem_p03436()
