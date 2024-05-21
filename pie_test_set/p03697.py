def problem_p03697(input_data):
    import sys

    stdin = sys.stdin

    import numpy as np

    ns = lambda: stdin.readline().rstrip()

    ni = lambda: int(ns())

    na = lambda: list(map(int, stdin.readline().split()))

    s = lambda h: [list(map(int, stdin.readline().split())) for i in range(h)]

    a, b = na()

    return a + b if a + b < 10 else "error"
