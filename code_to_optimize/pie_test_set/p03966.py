def problem_p03966():
    # -*- coding: utf-8 -*-

    import math

    import sys

    import itertools

    import numpy as np

    import functools

    import collections

    mo = 1000000007

    r = range

    n = int(eval(input()))

    x, y = list(map(int, input().split()))

    for i in range(n - 1):

        a, b = list(map(int, input().split()))

        k = max((x - 1) // a + 1, (y - 1) // b + 1)

        x, y = k * a, k * b

    print((x + y))


problem_p03966()
