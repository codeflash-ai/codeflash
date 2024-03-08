def problem_p02864():
    #!/usr/bin/env python3

    import sys

    import math

    from bisect import bisect_right as br

    from bisect import bisect_left as bl

    sys.setrecursionlimit(2147483647)

    from heapq import heappush, heappop, heappushpop

    from collections import defaultdict

    from itertools import accumulate

    from collections import Counter

    from collections import deque

    from operator import itemgetter

    from itertools import permutations

    mod = 10**9 + 7

    inf = float("inf")

    def I():
        return int(sys.stdin.readline())

    def LI():
        return list(map(int, sys.stdin.readline().split()))

    n, k = LI()

    h = [0] + LI()

    dp = [[inf] * (n - k + 1) for _ in range(n + 1)]

    dp[0][0] = 0

    for i in range(1, n + 1):

        for j in range(1, n - k + 1):

            for l in range(i):

                dp[i][j] = min(dp[i][j], dp[l][j - 1] + max(0, h[i] - h[l]))

    ans = inf

    for i in range(n + 1):

        ans = min(ans, dp[i][n - k])

    print(ans)


problem_p02864()
