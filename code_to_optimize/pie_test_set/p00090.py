def problem_p00090():
    import math, cmath

    e = 1e-6

    def f(m):

        c = 1j * math.acos(a / 2)

        p1 = cmath.exp(b + c) + p

        p2 = cmath.exp(b - c) + p

        s1, s2 = 2, 2

        for k in N:

            if k in [i, i1]:
                continue

            if abs(X[k] - p1) < 1 + e:
                s1 += 1

            if abs(X[k] - p2) < 1 + e:
                s2 += 1

        return max(m, s1, s2)

    while 1:

        n = eval(input())

        if n == 0:
            break

        N = list(range(n))

        X = []

        for i in N:

            x, y = list(map(float, eval(input())))

            X += [x + 1j * y]

        m = 1

        for i in N:

            p = X[i]

            for i1 in range(i + 1, n):

                a, b = cmath.polar(X[i1] - p)

                b *= 1j

                if a <= 2:
                    m = f(m)

        print(m)


problem_p00090()
