def problem_p00012():
    def S(x1, y1, x2, y2, x3, y3):

        return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0

    while True:

        try:

            x1, y1, x2, y2, x3, y3, xp, yp = list(map(float, input().split()))

            flag = (
                S(x1, y1, x2, y2, xp, yp)
                + S(x1, y1, x3, y3, xp, yp)
                + S(x3, y3, x2, y2, xp, yp)
                - S(x1, y1, x2, y2, x3, y3)
                > 0.0000001
            )

            print("NO" if flag else "YES")

        except:

            break


problem_p00012()
