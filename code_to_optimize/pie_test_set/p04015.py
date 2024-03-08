def problem_p04015():
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

    N, A = list(map(int, input().split()))

    X = list(map(int, input().split()))

    dp = [[0] * 2501 for _ in range(51)]

    dp[0][0] = 1

    for i in range(N):

        x = X[i]

        for j in range(i, -1, -1):

            for k in range(2501):

                if k < x:

                    continue

                dp[j + 1][k] += dp[j][k - x]

    ans = 0

    for i in range(1, 51):

        ans += dp[i][i * A]

    print(ans)


problem_p04015()
