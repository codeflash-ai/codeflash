def problem_p02658():
    """

    ~~ Author : Bhaskar

    ~~ Dated : 31~05~2020

    """

    import sys

    from bisect import *

    from math import floor, sqrt, ceil, factorial as F, gcd, pi

    from itertools import chain, combinations, permutations, accumulate

    from collections import Counter, defaultdict, OrderedDict, deque

    from array import array

    INT_MAX = sys.maxsize

    INT_MIN = -(sys.maxsize) - 1

    mod = 10**18

    lcm = lambda a, b: (a * b) // gcd(a, b)

    setbit = lambda x: bin(x)[2:].count("1")

    def solve():

        n = int(sys.stdin.readline())

        a = list(map(int, sys.stdin.readline().split()))

        ans = 1

        flag = False

        for i in a:

            ans *= i

            # print(ans)

            if ans > mod:

                flag = True

                break

        if 0 in a:

            print((0))

        else:

            if not flag:

                print(ans)

            else:

                print((-1))

    if __name__ == "__main__":

        solve()


problem_p02658()
