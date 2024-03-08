def problem_p00004():
    import sys

    for s in sys.stdin:

        a, b, c, d, e, f = list(map(float, s.split()))

        x = c * e - b * f

        y = a * f - c * d

        z = a * e - b * d

        def f(x, z):

            if x == 0:
                return 0

            else:
                return x / z

        print("%.3f %.3f" % (f(x, z), f(y, z)))


problem_p00004()
