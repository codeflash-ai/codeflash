def problem_p02920():
    from collections import defaultdict, deque, Counter

    from heapq import heappush, heappop, heapify

    import math

    import bisect

    import random

    from itertools import permutations, accumulate, combinations, product

    import sys

    import string

    from bisect import bisect_left, bisect_right

    from math import factorial, ceil, floor

    from operator import mul

    from functools import reduce

    sys.setrecursionlimit(2147483647)

    INF = 10**20

    def LI():
        return list(map(int, sys.stdin.buffer.readline().split()))

    def I():
        return int(sys.stdin.buffer.readline())

    def LS():
        return sys.stdin.buffer.readline().rstrip().decode("utf-8").split()

    def S():
        return sys.stdin.buffer.readline().rstrip().decode("utf-8")

    def IR(n):
        return [I() for i in range(n)]

    def LIR(n):
        return [LI() for i in range(n)]

    def SR(n):
        return [S() for i in range(n)]

    def LSR(n):
        return [LS() for i in range(n)]

    def SRL(n):
        return [list(S()) for i in range(n)]

    def MSRL(n):
        return [[int(j) for j in list(S())] for i in range(n)]

    mod = 1000000007

    from collections import defaultdict, deque, Counter

    from heapq import heappush, heappop, heapify

    import math

    import bisect

    import random

    from itertools import permutations, accumulate, combinations

    import sys

    import string

    from bisect import bisect_left

    INF = float("inf")

    def LI():
        return list(map(int, sys.stdin.readline().split()))

    def I():
        return int(sys.stdin.readline())

    def LIM():
        return list([int(x) - 1 for x in sys.stdin.readline().split()])

    def LS():
        return sys.stdin.readline().split()

    def S():
        return sys.stdin.readline().strip()

    def IR(n):
        return [I() for i in range(n)]

    def LIR(n):
        return [LI() for i in range(n)]

    def LIRM(n):
        return [LIM() for i in range(n)]

    def SR(n):
        return [S() for i in range(n)]

    def LSR(n):
        return [LS() for i in range(n)]

    def SRL(n):
        return [list(S()) for i in range(n)]

    mod = 1000000007

    n = I()

    final_slime = sorted(LI(), reverse=True)

    now_slime = [final_slime[0]]

    made_or_not = [0] * (2**n)

    made_or_not[0] = 1

    for i in range(n):

        k = 0

        for j in range(len(now_slime)):

            flag = 1

            while flag:

                if k == 2**n:

                    print("No")

                    exit(0)

                if made_or_not[k] == 0 and now_slime[j] - 1 >= final_slime[k]:

                    now_slime += [final_slime[k]]

                    made_or_not[k] = 1

                    flag = 0

                k += 1

        now_slime.sort(reverse=True)

    print("Yes")


problem_p02920()
