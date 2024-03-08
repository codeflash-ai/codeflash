def problem_p02578():
    # coding: utf-8

    # Your code here!

    # import math

    # from decimal import Decimal

    # from collections import deque, Counter

    # from itertools import product

    def LI():

        return list(map(int, input().split()))

    def OPEN():

        return list(map(int, open(0).read().split()))

    n, *a = OPEN()

    now = a[0]

    ans = []

    for i in range(n):

        if a[i] < now:

            ans += ((now - a[i]),)

            a[i] = now

        now = a[i]

    print((sum(ans)))


problem_p02578()
