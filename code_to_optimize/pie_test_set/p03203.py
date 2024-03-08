def problem_p03203():
    h, w, n = list(map(int, input().split()))

    xy = sorted([list(map(int, input().split())) for _ in range(n)])

    dx = 0

    dy = 0

    for x, y in xy:

        x -= dx

        y -= dy

        if x == y:

            dx += x - 1

            dy += y - 2

        elif y < x:

            print((dx + x - 1))

            break

    else:

        print(h)


problem_p03203()
