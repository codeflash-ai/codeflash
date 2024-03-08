def problem_p00143():
    def side(a, b, c):

        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0]) > 0

    def isInner(x):
        return side(p0, p1, x) == side(p1, p2, x) == side(p2, p0, x)

    for _ in [0] * eval(input()):

        P = list(map(int, input().split()))

        p0 = P[0:2]

        p1 = P[2:4]

        p2 = P[4:6]

        x1 = P[6:8]

        x2 = P[8:]

        print(["NG", "OK"][isInner(x1) != isInner(x2)])


problem_p00143()
