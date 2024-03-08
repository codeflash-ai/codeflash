def problem_p00122():
    dX = [2, 2, 2, 1, 0, -1, -2, -2, -2, -1, 0, 1]

    dY = [-1, 0, 1, 2, 2, 2, 1, 0, -1, -2, -2, -2]

    def solve(x, y, XY):

        xy = XY[:]

        if len(xy) == 0:
            return "OK"

        sx, sy = xy.pop(0), xy.pop(0)

        for dx, dy in zip(dX, dY):

            if doa(x + dx, y + dy, sx, sy):

                r = solve(x + dx, y + dy, xy)

                if r:
                    return r

    def doa(x, y, sx, sy):

        if not (0 <= x <= 9 and 0 <= y <= 9):
            return False

        return True if abs(x - sx) < 2 and abs(y - sy) < 2 else False

    while 1:

        x, y = list(map(int, input().split()))

        if x == y == 0:
            break

        n = eval(input())

        xy = list(map(int, input().split()))

        ans = solve(x, y, xy)

        print(ans if ans else "NA")


problem_p00122()
