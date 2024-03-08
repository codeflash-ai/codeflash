def problem_p03157():
    import sys

    # import math, string, itertools, fractions, heapq, collections, re, array, bisect, copy, functools, random

    from collections import deque, defaultdict, Counter
    from heapq import heappush, heappop

    # from itertools import permutations, combinations, product, accumulate, groupby

    # from bisect import bisect_left, bisect_right, insort_left, insort_right

    # from operator import itemgetter as ig

    sys.setrecursionlimit(10**7)

    inf = 10**20
    INF = float("INF")
    ans = 0
    tmp = 0
    ansli = []
    tmpli = []
    candili = []
    mod = 10**9 + 7

    dd = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    ddn = dd + [(-1, 1), (1, 1), (1, -1), (-1, -1)]
    ddn9 = ddn + [(0, 0)]

    """for dx, dy in dd:
    
            nx = j + dx; ny = i + dy
    
                if 0 <= nx < w and 0 <= ny < h:"""

    def wi():
        return list(map(int, sys.stdin.readline().split()))

    def wip():
        return [int(x) - 1 for x in sys.stdin.readline().split()]  # WideIntPoint

    def ws():
        return sys.stdin.readline().split()

    def i():
        return int(sys.stdin.readline())

    def s():
        return eval(input())

    def hi(n):
        return [i() for _ in range(n)]

    def hs(n):
        return [s() for _ in range(n)]  # HeightString

    def mi(n):
        return [wi() for _ in range(n)]  # MatrixInt

    def mip(n):
        return [wip() for _ in range(n)]

    def ms(n):
        return [ws() for _ in range(n)]

    # if __name__ == '__main__':

    h, w = wi()

    s = hs(h)

    flagli = [[0] * w for i in range(h)]

    stack = deque()

    black = 0

    white = 0

    for i in range(h):

        for j in range(w):

            if flagli[i][j] == 0:

                flagli[i][j] = 1

                white = 1 if s[i][j] == "." else 0

                black = 1 if s[i][j] == "#" else 0

                stack.append([i, j])

                while stack != deque():

                    p, q = stack.pop()

                    k = 0 if s[p][q] == "." else 1

                    for dx, dy in dd:

                        nx = q + dx

                        ny = p + dy

                        if 0 <= nx < w and 0 <= ny < h:

                            if flagli[ny][nx] == 0:

                                if k == 0 and s[ny][nx] == "#":

                                    flagli[ny][nx] = 1

                                    stack.append([ny, nx])

                                    black += 1

                                elif k == 1 and s[ny][nx] == ".":

                                    flagli[ny][nx] = 1

                                    stack.append([ny, nx])

                                    white += 1

                ans += black * white

                black = 0

                white = 0

    print(ans)


problem_p03157()
