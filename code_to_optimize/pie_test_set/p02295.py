def problem_p02295():
    def cp(a, b, c, d, e, f, g, h):

        A = a * d - b * c

        B = e * h - f * g

        C = d - b

        D = e - g

        E = c - a

        F = f - h

        det = C * D - E * F

        x = A * D + B * E

        y = A * F + B * C

        return [x / det, y / det]

    q = int(eval(input()))

    for i in range(q):

        a, b, c, d, e, f, g, h = [int(i) for i in input().split()]

        P = cp(a, b, c, d, e, f, g, h)

        print((P[0], P[1]))


problem_p02295()
