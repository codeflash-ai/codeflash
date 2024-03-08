def problem_p00163():
    # Highway Toll

    PRICE_LIST = (
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 300, 500, 600, 700, 1350, 1650),
        (0, 6, 0, 350, 450, 600, 1150, 1500),
        (0, 13, 7, 0, 250, 400, 1000, 1350),
        (0, 18, 12, 5, 0, 250, 850, 1300),
        (0, 23, 17, 10, 5, 0, 600, 1150),
        (0, 43, 37, 30, 25, 20, 0, 500),
        (0, 58, 52, 45, 40, 35, 15, 0),
    )

    while True:

        i = int(input())

        if i == 0:

            break

        it = int(input().strip().replace(" ", ""))

        o = int(input())

        ot = int(input().strip().replace(" ", ""))

        half = False

        if (1730 <= it <= 1930 or 1730 <= ot <= 1930) and PRICE_LIST[o][i] <= 40:

            half = True

        p = PRICE_LIST[i][o]

        if half:

            p /= 2

            if p % 50:

                p = (p / 50) * 50 + 50

        print(p)


problem_p00163()
