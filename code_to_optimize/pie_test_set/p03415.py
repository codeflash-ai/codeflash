def problem_p03415():
    import sys

    import numpy as np

    stdin = sys.stdin

    ri = lambda: int(rs())

    rl = lambda: list(map(int, stdin.readline().split()))

    rs = lambda: stdin.readline().rstrip()  # ignore trailing spaces

    C = [rs() for _ in range(3)]

    answer = C[0][0] + C[1][1] + C[2][2]

    print(answer)


problem_p03415()
