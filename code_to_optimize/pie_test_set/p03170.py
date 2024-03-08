def problem_p03170():
    # -*- coding: utf-8 -*-

    """

    Created on Sat Apr 25 18:20:35 2020

    """

    import sys

    import numpy as np

    sys.setrecursionlimit(10**9)

    def input():

        return sys.stdin.readline()[:-1]

    mod = 10**9 + 7

    # N = int(input())

    N, K = list(map(int, input().split()))

    A = np.array(list(map(int, input().split())))

    # N, K, *A = map(int, open(0).read().split())

    dp = [True for i in range(K + A[-1] + 1)]

    for i in range(K):

        if dp[i]:

            for a in A:

                dp[i + a] = False

    #    print(i, dp[i],A+i,dp[A+i])

    # print(dp)

    if dp[K]:

        ans = "Second"

    else:

        ans = "First"

    print(ans)


problem_p03170()
