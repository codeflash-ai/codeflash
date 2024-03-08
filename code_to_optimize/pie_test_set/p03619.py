def problem_p03619():
    import sys

    input = sys.stdin.readline

    from bisect import bisect_left

    from math import pi

    sx, sy, gx, gy = list(map(int, input().split()))

    N = int(eval(input()))

    XY = [tuple(int(x) for x in input().split()) for _ in range(N)]

    if sx > gx:

        sx, gx = gx, sx

        sy, gy = gy, sy

    gx -= sx

    gy -= sy

    XY = [(x - sx, y - sy) for x, y in XY if 0 <= x - sx <= gx]

    if gy >= 0:

        XY = [(x, y) for x, y in XY if 0 <= y <= gy]

    else:

        gy = -gy

        XY = [(x, -y) for x, y in XY if 0 <= -y <= gy]

    # xは相異なるので...

    XY.sort()

    Y = [y for x, y in XY]

    # longest increasing subseq

    INF = 10**18

    dp = [INF] * (len(Y) + 1)

    for y in Y:

        idx = bisect_left(dp, y)

        dp[idx] = y

    L = bisect_left(dp, INF)

    if L <= min(gx, gy):

        answer = 100 * (gx + gy) + (10 * pi / 2 - 20) * L

    else:

        answer = 100 * (gx + gy) + (10 * pi / 2 - 20) * (L - 1) + (10 * pi - 20)

    print(answer)


problem_p03619()
