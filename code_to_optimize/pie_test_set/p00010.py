def problem_p00010():
    def calc(a, b, c, d, e, f):

        A = 2 * (c - a)

        B = 2 * (d - b)

        C = a * a - c * c + b * b - d * d

        D = 2 * (e - a)

        E = 2 * (f - b)

        F = a * a - e * e + b * b - f * f

        N = A * E - D * B

        X = (B * F - E * C) / N

        Y = (C * D - F * A) / N

        R = ((X - a) ** 2 + (Y - b) ** 2) ** 0.5

        return tuple(map(round, [X, Y, R], [3] * 3))

    l = [list(map(float, input().split())) for i in range(eval(input()))]

    for k in l:
        print("%.3f %.3f %.3f" % (calc(*k)))


problem_p00010()
