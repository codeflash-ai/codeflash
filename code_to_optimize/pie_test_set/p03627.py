def problem_p03627():
    # import bisect

    from collections import Counter, deque

    # from copy import copy, deepcopy

    # from fractions import gcd

    # from functools import reduce

    # from itertools import accumulate, permutations, combinations, combinations_with_replacement, groupby, product

    # import math

    # import numpy as np

    # from operator import xor

    import sys

    sys.setrecursionlimit(10**5 + 10)

    # input = sys.stdin.readline

    def resolve():

        N = int(eval(input()))

        A = list(map(int, input().split()))

        AA = Counter(A).most_common()

        ls = []

        for i in range(len(AA)):

            if AA[i][1] >= 2:

                ls.append(AA[i])

            else:

                break

        ls = sorted(ls, key=lambda x: x[0])

        try:

            ans = ls[-1][0] * ls[-2][0]

            for i in reversed(list(range(len(ls)))):

                if ls[i][1] >= 4:

                    ans = max(ans, ls[i][0] ** 2)

            print(ans)

        except:

            print((0))

    resolve()


problem_p03627()
