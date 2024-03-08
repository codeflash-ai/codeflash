def problem_p02841():
    import sys

    import numpy as np

    sys.setrecursionlimit(10**9)

    stdin = sys.stdin

    ri = lambda: int(rs())

    rl = lambda: list(map(int, stdin.readline().split()))

    rs = lambda: stdin.readline().rstrip()  # ignore trailing spaces

    M1, D1 = rl()

    M2, D2 = rl()

    print((1 if M1 != M2 else 0))


problem_p02841()
