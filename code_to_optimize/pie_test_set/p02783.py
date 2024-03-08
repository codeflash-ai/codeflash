def problem_p02783():
    import collections

    import itertools as it

    import math

    # import numpy as np

    #  = input()

    #  = int(input())

    h, a = list(map(int, input().split()))

    #  = list(map(int, input().split()))

    #  = [int(input()) for i in range(N)]

    #

    # c = collections.Counter()

    if h % a == 0:

        print((int(h / a)))

    else:

        print((h // a + 1))


problem_p02783()
