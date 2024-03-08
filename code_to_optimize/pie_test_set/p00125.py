def problem_p00125():
    cal = {}

    month = [
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    ]

    count = 1

    for y in range(3000):  #

        f = 1 if y % 4 == 0 and y % 100 != 0 or y % 400 == 0 else 0

        for m in range(12):

            for d in range(month[f][m]):

                cal[(y, m + 1, d + 1)] = count

                count += 1

    while True:

        y1, m1, d1, y2, m2, d2 = list(map(int, input().split()))

        if any(i < 0 for i in (y1, m1, d1, y2, m2, d2)):

            break

        print(cal[(y2, m2, d2)] - cal[(y1, m1, d1)])


problem_p00125()
