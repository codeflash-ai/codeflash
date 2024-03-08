def problem_p00035():
    import sys

    def sign(x):

        if x > 0:
            s = 1

        elif x == 0:
            s = 0

        else:
            s = -1

        return s

    def f(p1, p2, p3):

        s = sign((p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))

        return s

    for s in sys.stdin:

        D = list(map(float, s.split(",")))

        p1 = D[0:2]
        p2 = D[2:4]
        p3 = D[4:6]
        p4 = D[6:8]

        x = f(p1, p3, p2) != f(p1, p3, p4) and f(p2, p4, p1) != f(p2, p4, p3)

        print(["NO", "YES"][x])


problem_p00035()
