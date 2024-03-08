def problem_p02639():
    from sys import stdin, stdout

    from math import gcd, sqrt

    from collections import deque

    input = stdin.readline

    R = lambda: list(map(int, input().split()))

    I = lambda: int(eval(input()))

    S = lambda: input().rstrip("\n")

    L = lambda: list(R())

    P = lambda x: stdout.write(x)

    hg = lambda x, y: ((y + x - 1) // x) * x

    pw = lambda x: 1 if x == 1 else 1 + pw(x // 2)

    chk = lambda x: chk(x // 2) if not x % 2 else True if x == 1 else False

    print((list(map(int, input().split())).index(0) + 1))


problem_p02639()
