def problem_p02973():
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

    def main():

        n = I()

        a = [I() for _ in range(n)]

        dp = [inf] * n

        for i in range(n):

            r = br(dp, -a[i])

            dp[r] = -a[i]

        print((n - dp.count(inf)))

        return

    if __name__ == "__main__":

        main()


problem_p02973()
