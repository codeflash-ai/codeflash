def problem_p00454():
    import sys

    sys.setrecursionlimit(100000000)

    while True:

        w, h = list(map(int, input().split()))

        if not w:

            break

        n = int(eval(input()))

        xlst = [0, w - 1]

        ylst = [0, h - 1]

        plst = []

        for i in range(n):

            x1, y1, x2, y2 = list(map(int, input().split()))

            plst.append([x1, y1, x2 - 1, y2 - 1])

            xlst.append(x1)

            #    xlst.append(x1 + 1)

            xlst.append(x2)

            xlst.append(x2 - 1)

            ylst.append(y1)

            #    ylst.append(y1 + 1)

            ylst.append(y2)

            ylst.append(y2 - 1)

        xlst = list(set(xlst))

        ylst = list(set(ylst))

        sorted_xlst = sorted(xlst)

        sorted_ylst = sorted(ylst)

        xdic = {}

        ydic = {}

        for i, v in enumerate(sorted_xlst):

            xdic[v] = i

        for i, v in enumerate(sorted_ylst):

            ydic[v] = i

        neww = xdic[sorted_xlst[-1]]

        newh = ydic[sorted_ylst[-1]]

        #  print(neww, newh)

        painted = [[0] * (newh) for _ in range(neww)]

        def paint_area(x, y):

            painted[x][y] = 1

            for tx, ty in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:

                if 0 <= tx <= neww - 1 and 0 <= ty <= newh - 1 and not painted[tx][ty]:

                    paint_area(tx, ty)

        for p in plst:

            x1, y1, x2, y2 = p

            x1, y1, x2, y2 = xdic[x1], ydic[y1], xdic[x2], ydic[y2]

            for x in range(x1, x2 + 1):

                for y in range(y1, y2 + 1):

                    painted[x][y] = 1

        #  for area in painted:

        #    print(area)

        #  print()

        ans = 0

        for x in range(neww):

            for y in range(newh):

                if not painted[x][y]:

                    ans += 1

                    painted[x][y] = 1

                    que = [(x, y)]

                    while que:

                        px, py = que.pop()

                        for tx, ty in [(px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)]:

                            if 0 <= tx <= neww - 1 and 0 <= ty <= newh - 1 and not painted[tx][ty]:

                                painted[tx][ty] = 1

                                que.append((tx, ty))

        print(ans)


problem_p00454()
