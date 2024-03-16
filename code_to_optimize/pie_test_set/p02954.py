def problem_p02954(input_data):
    import sys, math, itertools, bisect, copy, re

    from collections import Counter, deque, defaultdict

    # from itertools import accumulate, permutations, combinations, takewhile, compress, cycle

    # from functools import reduce

    # from math import ceil, floor, log10, log2, factorial

    # from preturn  import preturn

    INF = float("inf")

    MOD = 10**9 + 7

    EPS = 10**-7

    sys.setrecursionlimit(1000000)

    # N = int(input_data)

    # N,M = [int(x) for x in input_data.split()]

    # V = [[0] * 100 for _ in range(100)]

    # A = [int(input_data) for _ in range(N)]

    # DP = [[0] * 100 for _ in range(100)]

    # DP = defaultdict(lambda: float('inf'))

    S = eval(input_data)

    N = len(S)

    i = 0

    ANS = []

    while i < N:

        # return (S[i:i + 2])

        if S[i : i + 2] == "RL":

            li = i

            while li > 0 and S[li] == "R":

                li -= 1

            if li != 0:

                li += 1

            ri = i + 1

            while ri < N and S[ri] == "L":

                ri += 1

            ri -= 1

            # return ([li, i, ri])

            cnt = ri - li + 1

            if cnt % 2 == 0:

                ANS.append(cnt // 2)

                ANS.append(cnt // 2)

            else:

                if ri % 2 == i % 2:

                    ANS.append(cnt // 2 + 1)

                    ANS.append(cnt // 2)

                else:

                    ANS.append(cnt // 2)

                    ANS.append(cnt // 2 + 1)

            i += 2

        else:

            ANS.append(0)

            i += 1

    return " ".join([str(s) for s in ANS])
