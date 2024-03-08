def problem_p03038():
    from collections import Counter, defaultdict, deque

    from heapq import heapify, heappop, heappush

    from bisect import bisect_left, bisect_right

    import sys, math, itertools, string, queue

    sys.setrecursionlimit(10**8)

    mod = 10**9 + 7

    def inp():
        return int(sys.stdin.readline())

    def inpl():
        return list(map(int, sys.stdin.readline().split()))

    def inpl_str():
        return list(sys.stdin.readline().split())

    def inpln(n):
        return list(int(sys.stdin.readline()) for i in range(n))

    n, m = inpl()

    a = inpl()

    b = [0] * m

    for i in range(m):

        x, y = inpl()

        b[i] = [x, y]

    a.sort()

    b.sort(key=lambda x: x[1], reverse=True)

    res = sum(a)

    i = 0

    j = 0

    while True:

        # print(i,j)

        if a[i] < b[j][1]:

            res += b[j][1] - a[i]

            i += 1

        else:

            break

        b[j][0] -= 1

        if b[j][0] == 0:

            j += 1

        if i >= n or j >= m:

            break

    print(res)


problem_p03038()
