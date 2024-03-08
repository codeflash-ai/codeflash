def problem_p02700():
    import sys

    input = sys.stdin.readline

    from collections import *

    A, B, C, D = list(map(int, input().split()))

    while True:

        C -= B

        if C <= 0:

            print("Yes")

            exit()

        A -= D

        if A <= 0:

            print("No")

            exit()


problem_p02700()
