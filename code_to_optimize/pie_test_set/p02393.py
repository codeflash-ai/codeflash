def problem_p02393():
    x = input().split()

    y = list(map(int, x))

    a = y[0]

    b = y[1]

    c = y[2]

    d = sorted([a, b, c])

    print(("{0} {1} {2}".format(d[0], d[1], d[2])))


problem_p02393()
