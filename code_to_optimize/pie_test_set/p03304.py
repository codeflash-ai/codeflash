def problem_p03304():
    # -*- coding: utf-8 -*-

    from math import log2

    line = list(map(int, input().split(" ")))

    n = line[0]

    m = line[1]

    d = line[2]

    if 2 * d < n and d > 0:

        ret1 = 1 + (m - 2) * log2(n)

        ret2 = log2(n - d)

        ret3 = log2(m - 1)

        ret4 = m * log2(n)

        print((2 ** (ret1 + ret2 + ret3 - ret4)))

    else:

        ret1 = (m - 1) * log2(n)

        ret2 = log2(m - 1)

        ret3 = m * log2(n)

        print((2 ** (ret1 + ret2 - ret3)))


problem_p03304()
