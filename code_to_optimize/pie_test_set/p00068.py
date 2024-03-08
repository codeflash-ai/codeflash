def problem_p00068():
    import sys

    def side(p1, p2):

        global D

        y1, x1 = p1

        y2, x2 = p2

        dy = y2 - y1

        dx = x2 - x1

        for p3 in D:

            if p1 != p3 and p2 != p3 and (p3[1] - x1) * dy - dx * (p3[0] - y1) < 0:
                return 0

        else:
            return 1

    while 1:

        n = eval(input())

        if n == 0:
            break

        D = sorted([list(eval(input())) for i in range(n)])

        p = p1 = D[0]

        while 1:

            for p2 in D:

                if p1 != p2 and side(p1, p2):
                    break

            p1 = p2

            D.remove(p2)

            if p2 == p:
                break

        print(len(D))


problem_p00068()
