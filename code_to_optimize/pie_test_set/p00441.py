def problem_p00441():
    p = 100

    while 1:

        n = eval(input())

        if n == 0:
            break

        xy = sorted([list(map(int, input().split())) for i in range(n)])

        S = set(map(tuple, xy))

        ans = 0

        for i in range(n):

            x1, y1 = xy[i]

            xy = xy[:i] + sorted(xy[i:], key=lambda XY: (XY[0] - x1) ** 2 + (XY[1] - y1) ** 2)

            cur = 0

            for j in range(n - 1, i, -1):

                x2, y2 = xy[j]

                a = (x2 - y2 + y1, y2 + x2 - x1)

                b = (x1 - y2 + y1, y1 + x2 - x1)

                if a in S and b in S:

                    cur = (x1 - x2) ** 2 + (y1 - y2) ** 2

                    break

            ans = max(ans, cur)

        print(ans)


problem_p00441()
