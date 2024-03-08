def problem_p03240():
    N = int(eval(input()))

    xn = [0] * N

    yn = [0] * N

    hn = [0] * N

    xyh = [[int(j) for j in input().split()] for i in range(N)]

    div = 0

    for Cx in range(101):

        for Cy in range(101):

            for i in range(N):

                xi, yi, hi = xyh[i]

                if 0 < hi:

                    h = abs(xi - Cx) + abs(yi - Cy) + hi

                    if div in [0, 2]:

                        H = h

                        div = 1

                    else:

                        if H != h:

                            div = 2

                            break

            if div == 2:

                continue

            for i in range(N):

                xi, yi, hi = xyh[i]

                if 0 == hi:

                    h = abs(xi - Cx) + abs(yi - Cy)

                    if h < H:

                        div = 2

                        break

            if div == 2:

                continue

            print((Cx, Cy, H))

            break

        if div == 1:

            break


problem_p03240()
