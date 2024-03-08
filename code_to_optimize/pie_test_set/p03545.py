def problem_p03545():

    # import numpy as np

    # import numpypy as np

    import sys

    input = sys.stdin.readline

    def eprint(*args, **kwargs):

        print(*args, file=sys.stderr, **kwargs)

        return

    import math

    import string

    import fractions

    from fractions import Fraction

    from fractions import gcd

    def lcm(n, m):

        return int(n * m / gcd(n, m))

    import re

    import array

    import copy

    import functools

    import operator

    import collections

    import itertools

    import bisect

    import heapq

    from heapq import heappush

    from heapq import heappop

    from heapq import heappushpop

    from heapq import heapify

    from heapq import heapreplace

    from queue import PriorityQueue as pq

    def reduce(p, q):

        common = fractions.gcd(p, q)

        return (p // common, q // common)

    # from itertools import accumulate

    # from collections import deque

    import random

    def main():

        # l = list(map(int, input().split()))

        ll = input().strip()

        l = [0 for i in range(4)]

        for i in range(len(l)):

            l[i] = int(ll[i])

        NUM_CASE = 3

        for case in range(2**NUM_CASE):

            # eprint("case : " + str(case))

            sum = l[0]

            stack = []

            S = ""

            for index in range(NUM_CASE):

                if (case >> index) & 1 == 1:

                    # eprint("l[%d] : %d" % (index+1,l[index+1]) )

                    sum += l[index + 1]

                    stack.append("+")

                else:

                    # eprint("l[%d] : %d" % (index+1,l[index+1]) )

                    sum -= l[index + 1]

                    stack.append("-")

            # eprint("sum : " + str(sum))

            if sum == 7:

                for i in range(4):

                    S += str(l[i])

                    if i != 3:

                        S += str(stack[i])

                S += "=7"

                print(S)

                return

        return

    if __name__ == "__main__":

        main()


problem_p03545()
