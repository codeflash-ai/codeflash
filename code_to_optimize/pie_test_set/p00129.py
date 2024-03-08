def problem_p00129():
    import math as M

    def R(A):
        return (A[0] ** 2 + A[1] ** 2) ** 0.5

    def I(i):
        return [list(map(int, input().split())) for _ in [0] * i]

    def C(a, b):
        return a > b or abs(a - b) < 1e-6

    def f(e1):

        tx, ty, sx, sy = e1

        x = []

        for e2 in WP:

            wx, wy, r = e2

            wt = [tx - wx, ty - wy]
            rwt = R(wt)

            sw = [wx - sx, wy - sy]
            rsw = R(sw)

            st = [tx - sx, ty - sy]
            rst = R(st)

            F = [rwt < r, rsw < r]

            if rst == 0:
                c = 1

            elif F == [1, 1]:
                c = 1

            elif F == [1, 0] or F == [0, 1]:
                c = 0

            elif F == [0, 0]:

                a = M.pi / 2 - M.acos(r / rsw)

                b = M.acos(round((sw[0] * st[0] + sw[1] * st[1]) / rsw / rst, 4))

                if C(a, b) and C(rst**2, rsw**2 - r**2):
                    c = 0

                else:
                    c = 1

            if c == 0:
                return 0

            x.append(c)

        return all(x)

    while 1:

        n = eval(input())

        if n == 0:
            break

        WP = I(n)

        P = I(eval(input()))

        for e in P:
            print(["Safe", "Danger"][f(e)])


problem_p00129()
