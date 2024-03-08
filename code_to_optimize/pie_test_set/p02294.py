def problem_p02294():
    class Line:

        def __init__(self, p1, p2):

            if p1[1] < p2[1]:
                self.s = p2
                self.e = p1

            elif p1[1] > p2[1]:
                self.s = p1
                self.e = p2

            else:

                if p1[0] < p2[0]:
                    self.s = p1
                    self.e = p2

                else:
                    self.s = p2
                    self.e = p1

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def dif(a, b):
        return [x - y for x, y in zip(a, b)]

    def InterSection(l, m):

        a = dif(l.e, l.s)
        b = dif(m.e, l.s)
        c = dif(m.s, l.s)

        d = dif(m.e, m.s)
        e = dif(l.e, m.s)
        f = dif(l.s, m.s)

        g = lambda a, b: cross(a, b) == 0 and dot(a, b) > 0 and dot(b, b) < dot(a, a)

        if g(a, b) or g(a, c) or g(d, e) or g(d, f):
            return True

        elif l.s == m.e or l.s == m.s or l.e == m.e or l.e == m.s:
            return True

        elif cross(a, b) * cross(a, c) >= 0 or cross(d, e) * cross(d, f) >= 0:
            return False

        else:
            return True

    q = int(eval(input()))

    for i in range(q):

        x0, y0, x1, y1, x2, y2, x3, y3 = [int(i) for i in input().split()]

        a = [x0, y0]
        b = [x1, y1]
        c = [x2, y2]
        d = [x3, y3]

        l1 = Line(a, b)
        l2 = Line(c, d)

        if InterSection(l1, l2):
            print((1))

        else:
            print((0))


problem_p02294()
