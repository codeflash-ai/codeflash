def problem_p02700(input_data):
    import sys

    input = sys.stdin.readline

    from collections import *

    A, B, C, D = list(map(int, input_data.split()))

    while True:

        C -= B

        if C <= 0:

            return "Yes"

            exit()

        A -= D

        if A <= 0:

            return "No"

            exit()
