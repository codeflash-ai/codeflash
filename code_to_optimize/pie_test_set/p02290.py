def problem_p02290():
    import sys

    from itertools import starmap

    readline = sys.stdin.readline

    p0, p1 = starmap(complex, list(zip(*[list(map(int, readline().split()))] * 2)))

    n = int(readline())

    def cross(a, b):

        return a.real * b.imag - a.imag * b.real

    def dot(a, b):

        return a.real * b.real + a.imag * b.imag

    def norm(base):

        return base.real * base.real + base.imag * base.imag

    def project(p0, p1, p2):

        base = p0 - p1

        r = dot(p2 - p0, base) / norm(base)

        return p0 + base * r

    for _ in [0] * n:

        p2 = complex(*[int(i) for i in readline().split()])

        ap = project(p0, p1, p2)

        print((ap.real, ap.imag))


problem_p02290()
