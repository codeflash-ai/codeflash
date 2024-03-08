def problem_p00713():
    from math import sqrt

    from bisect import bisect_left

    def circle_center(x1, y1, x2, y2):

        xd = x2 - x1
        yd = y2 - y1

        d = xd**2 + yd**2

        k = sqrt((4.0 - d) / d) / 2.0

        xc = (x1 + x2) / 2.0

        yc = (y1 + y2) / 2.0

        return [[xc - k * yd, yc + k * xd], [xc + k * yd, yc - k * xd]]

    while 1:

        n = int(input())

        if n == 0:
            break

        p = [list(map(float, input().split())) for i in range(n)]

        p.sort()

        prev = 0

        ans = 1

        for i in range(n):

            bx, by = p[i]

            while bx - p[prev][0] >= 2.0:
                prev += 1

            for j in range(i + 1, n):

                cx, cy = p[j]

                if cx - bx >= 2.0:
                    break

                if (bx - cx) ** 2 + (by - cy) ** 2 <= 4.0:

                    for ex, ey in circle_center(bx, by, cx, cy):

                        count = 2

                        for k in range(prev, n):

                            if k == i or k == j:
                                continue

                            dx, dy = p[k]

                            if dx - bx >= 2.0:
                                break

                            if (ex - dx) ** 2 + (ey - dy) ** 2 <= 1.0:

                                count += 1

                        ans = max(ans, count)

        print(ans)


problem_p00713()
