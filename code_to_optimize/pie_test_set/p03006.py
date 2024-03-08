def problem_p03006():
    n = int(eval(input()))

    xy = [tuple(map(int, input().split())) for _ in range(n)]

    s_xy = set(xy)

    count = 0

    for i in range(n - 1):

        for j in range(i + 1, n):

            ix, iy = xy[i]

            jx, jy = xy[j]

            p = jx - ix

            q = jy - iy

            tc = sum((x - p, y - q) in xy for x, y in s_xy)

            count = max(count, tc)

    print((n - count))


problem_p03006()
