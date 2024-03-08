def problem_p03994():
    from collections import Counter, defaultdict, deque

    from heapq import heappop, heappush, heapify

    import sys, bisect, math, itertools, fractions, pprint

    sys.setrecursionlimit(10**8)

    mod = 10**9 + 7

    INF = float("inf")

    def inp():
        return int(sys.stdin.readline())

    def inpl():
        return list(map(int, sys.stdin.readline().split()))

    def inpln(n):
        return list(int(sys.stdin.readline()) for i in range(n))

    s = eval(input())

    n = len(s)

    r = [ord("z") - ord(t) + 1 if t != "a" else 0 for t in s]

    res = [None] * n

    k = inp()

    for i, x in enumerate(r):

        if x <= k:

            res[i] = "a"

            k -= x

    if k > 0:

        tmp = "a" if res[-1] != None else s[-1]

        res[-1] = chr((ord(tmp) - ord("a") + k) % 26 + ord("a"))

    for i, t in enumerate(res):

        if t == None:

            res[i] = s[i]

    print(("".join(res)))


problem_p03994()
