def problem_p03815():
    n = int(input())

    c = 2 * (n / 11)

    h = {0: 0}

    for i in range(1, 6 + 1):

        h[i] = 1

    for i in range(7, 12):

        h[i] = 2

    if n in h:

        print(h[n])

    else:

        print(h[n % 11] + c)


problem_p03815()
