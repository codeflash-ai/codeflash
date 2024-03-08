def problem_p03504():
    import sys

    import numpy as np

    input = sys.stdin.readline

    n, c = list(map(int, input().split()))

    tv_guide = np.zeros((c, 10**5 + 1), dtype=np.int)

    for _ in range(n):

        s, t, c = [int(x) - 1 for x in input().split()]

        tv_guide[c][s : t + 1] = 1

    ans = 0

    for time in range(10**5 + 1):

        count = sum(tv_guide[:, time])

        ans = max(count, ans)

    print(ans)


problem_p03504()
