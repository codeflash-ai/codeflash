def problem_p03973():
    n, *a = list(map(int, open(0)))

    b, c = 1, 0

    for i in a:

        if i > b:

            c += (i - 1) // b

            b += b < 2

        elif i == b:

            b += 1

    print(c)


problem_p03973()
