def problem_p04045():
    import sys

    import itertools

    # import numpy as np

    import time

    import math

    import heapq

    from collections import defaultdict

    sys.setrecursionlimit(10**7)

    INF = 10**18

    MOD = 10**9 + 7

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    readlines = sys.stdin.buffer.readlines

    # map(int, input().split())

    N, K = list(map(int, input().split()))

    D = input().split()

    for n in range(N, 100000):

        s = str(n)

        ok = True

        for c in s:

            if c in D:

                ok = False

        if ok:

            print(n)

            break


problem_p04045()
