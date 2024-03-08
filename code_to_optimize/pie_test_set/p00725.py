def problem_p00725():
    def rec(x, y, t):

        global ans

        if t >= ans:
            return

        if field[y][x] == 3:
            ans = min(ans, t)

        for dx, dy in zip([1, 0, -1, 0], [0, 1, 0, -1]):

            nx, ny = x + dx, y + dy

            while 0 <= nx < W and 0 <= ny < H:

                if field[ny][nx] == 3:

                    rec(nx, ny, t + 1)

                if field[ny][nx] == 1:

                    if abs(nx - x) + abs(ny - y) == 1:
                        break

                    field[ny][nx] = 0

                    rec(nx - dx, ny - dy, t + 1)

                    field[ny][nx] = 1

                    break

                nx += dx
                ny += dy

    while 1:

        W, H = list(map(int, input().split()))

        if W == 0:
            break

        field = [list(map(int, input().split())) for _ in range(H)]

        for y in range(H):

            for x in range(W):

                if field[y][x] == 2:
                    sx, sy = x, y

        ans = 11

        rec(sx, sy, 0)

        print(ans if ans <= 10 else -1)


problem_p00725()
