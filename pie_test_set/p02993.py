def problem_p02993(input_data):
    #!/usr/bin/env python3

    import math
    import sys

    input = lambda: sys.stdin.buffer.readline().rstrip().decode("utf-8")

    sys.setrecursionlimit(10**8)

    inf = float("inf")

    ans = count = 0

    S = eval(input_data)

    for i in range(3):

        if S[i] == S[i + 1]:

            return "Bad"

            exit()

    return "Good"
