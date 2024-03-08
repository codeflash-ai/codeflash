def problem_p02580():
    import sys

    from bisect import bisect_left as bl

    input = sys.stdin.readline

    H, W, N = list(map(int, input().split()))

    ys = [0] * (H + 1)

    xs = [0] * (W + 1)

    a = []

    base = 10**6

    for i in range(N):

        y, x = list(map(int, input().split()))

        ys[y] += 1

        xs[x] += 1

        a.append((y, x))

    sy = sorted(ys[1:])

    sx = sorted(xs[1:])

    # print(sy, sx)

    def check(k):

        res = 0

        for y, x in a:

            res -= (ys[y] + xs[x]) == k

            res += (ys[y] + xs[x]) > k

        # print(res, k)

        for y in sy:

            i = bl(sx, k - y)

            res += W - i

            # print(W - i, y, k, res)

        return res > 0

    ok = 0

    ng = N + 1

    while ng - ok > 1:

        m = (ok + ng) // 2

        if check(m):
            ok = m

        else:
            ng = m

    print(ok)


problem_p02580()
