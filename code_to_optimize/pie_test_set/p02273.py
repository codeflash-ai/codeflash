def problem_p02273():
    import math

    class Pos:

        def __init__(self, x, y):

            self.x = x

            self.y = y

        def __str__(self):

            return str(self.x) + " " + str(self.y)

    def koch_curve(i, n, start, end):

        if i < n:

            f = get_pos(start, end, 1 / 3)

            t = get_pos(start, end, 2 / 3)

            s = get_sec_pos(start, end, f, t)

            koch_curve(i + 1, n, start, f)

            print(f)

            koch_curve(i + 1, n, f, s)

            print(s)

            koch_curve(i + 1, n, s, t)

            print(t)

            koch_curve(i + 1, n, t, end)

    def get_pos(start, end, m):

        dX = (end.x - start.x) * m

        dY = (end.y - start.y) * m

        return Pos(start.x + dX, start.y + dY)

    def get_sec_pos(start, end, f, t):

        if f.y == t.y:

            x = start.x + (end.x - start.x) / 2

            len = math.fabs(t.x - f.x)

            h = math.sqrt(len**2 - (len / 2) ** 2)

            if start.x < end.x:

                y = start.y + h

            else:

                y = start.y - h

        elif f.x < t.x and f.y < t.y:

            x = start.x

            y = t.y

        elif f.x > t.x and f.y < t.y:

            x = end.x

            y = f.y

        elif f.x > t.x and f.y > t.y:

            x = start.x

            y = t.y

        else:

            x = end.x

            y = f.y

        return Pos(x, y)

    n = int(eval(input()))

    start = Pos(0, 0)

    end = Pos(100, 0)

    print(start)

    koch_curve(0, n, start, end)

    print(end)


problem_p02273()
