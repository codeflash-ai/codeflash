def problem_p03675():
    import sys

    import collections

    import math

    n = int(eval(input()))

    A = [int(x) for x in input().split()]

    R = [0] * n

    idx = 0

    for i in range(n - 1, -1, -2):

        R[idx] = A[i]

        idx += 1

    for i in range(n % 2, n, +2):

        R[idx] = A[i]

        idx += 1

    print((" ".join([str(i) for i in R])))

    sys.exit(0)


problem_p03675()
