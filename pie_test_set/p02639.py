def problem_p02639(input_data):
    from collections import deque
    from math import gcd, sqrt
    from sys import stdin, stdout

    input = stdin.readline

    R = lambda: list(map(int, input_data.split()))

    I = lambda: int(eval(input_data))

    S = lambda: input_data.rstrip("\n")

    L = lambda: list(R())

    P = lambda x: stdout.write(x)

    hg = lambda x, y: ((y + x - 1) // x) * x

    pw = lambda x: 1 if x == 1 else 1 + pw(x // 2)

    chk = lambda x: chk(x // 2) if not x % 2 else True if x == 1 else False

    return list(map(int, input_data.split())).index(0) + 1
