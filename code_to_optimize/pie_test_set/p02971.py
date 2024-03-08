def problem_p02971():

    def s0():
        return eval(input())

    def s1():
        return input().split()

    def s2(n):
        return [eval(input()) for x in range(n)]

    def s3(n):
        return [input().split() for _ in range(n)]

    def s4(n):
        return [[x for x in s] for s in s2(n)]

    def n0():
        return int(eval(input()))

    def n1():
        return [int(x) for x in input().split()]

    def n2(n):
        return [int(eval(input())) for _ in range(n)]

    def n3(n):
        return [[int(x) for x in input().split()] for _ in range(n)]

    def t3(n):
        return [tuple(int(x) for x in input().split()) for _ in range(n)]

    def p0(b, yes="Yes", no="No"):
        print((yes if b else no))

    # from sys import setrecursionlimit

    # setrecursionlimit(1000000)

    # from collections import Counter,deque,defaultdict

    # import itertools

    # import math

    # import networkx

    # from bisect import bisect_left,bisect_right

    # from heapq import heapify,heappush,heappop

    n = n0()

    a = n2(n)

    b = sorted(a)

    m1 = b[-1]

    m2 = b[-2]

    for i in a:

        p0(i == m1, m2, m1)


problem_p02971()
