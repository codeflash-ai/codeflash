def problem_p03502():
    """

    author : halo2halo

    date : 29, Jan, 2020

    """

    import sys

    # import itertools

    import numpy as np

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    readlines = sys.stdin.buffer.readlines

    sys.setrecursionlimit(10**7)

    def Hershad(n):

        if n == 0:

            return 0

        return n % 10 + Hershad(n // 10)

    N = int(readline())

    print(("Yes" if N % Hershad(N) == 0 else "No"))


problem_p03502()
