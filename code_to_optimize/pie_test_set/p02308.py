def problem_p02308():
    #!/usr/bin/env python

    # -*- coding: utf-8 -*-

    """

    input:

    2 1 1

    2

    0 1 4 1

    3 0 3 3



    output:

    1.00000000 1.00000000 3.00000000 1.00000000

    3.00000000 1.00000000 3.00000000 1.00000000

    """

    import sys

    import math

    class Segment(object):

        __slots__ = ("source", "target")

        def __init__(self, source, target):

            self.source = complex(source)

            self.target = complex(target)

    class Circle(object):

        __slots__ = ("centre", "radius")

        def __init__(self, x, y, r):

            self.centre = x + y * 1j

            self.radius = r

    def dot(a, b):

        return a.real * b.real + a.imag * b.imag

    def project(s, p):

        base_vector = s.target - s.source

        prj_ratio = dot(p - s.source, base_vector) / pow(abs(base_vector), 2)

        return s.source + base_vector * prj_ratio

    def get_cross_point(c, l):

        prj_vector = project(l, c.centre)

        line_unit_vector = (l.target - l.source) / (abs(l.target - l.source))

        base = math.sqrt(pow(c.radius, 2) - pow(abs(prj_vector - c.centre), 2))

        p1, p2 = prj_vector + line_unit_vector * base, prj_vector - line_unit_vector * base

        if p1.real < p2.real:

            ans = (p1, p2)

        elif p1.real == p2.real:

            if p1.imag < p2.imag:

                ans = (p1, p2)

            else:

                ans = (p2, p1)

        else:

            ans = (p2, p1)

        return ans

    def solve(_lines):

        for line in _lines:

            line_axis = tuple(map(int, line))

            p0, p1 = (x + y * 1j for x, y in zip(line_axis[::2], line_axis[1::2]))

            l = Segment(p0, p1)

            cp1, cp2 = get_cross_point(circle, l)

            print(
                ("{0:.8f} {1:.8f} {2:.8f} {3:.8f}".format(cp1.real, cp1.imag, cp2.real, cp2.imag))
            )

        return None

    if __name__ == "__main__":

        _input = sys.stdin.readlines()

        cx, cy, radius = list(map(int, _input[0].split()))

        q_num = int(_input[1])

        lines = [x.split() for x in _input[2:]]

        circle = Circle(cx, cy, radius)

        solve(lines)


problem_p02308()
