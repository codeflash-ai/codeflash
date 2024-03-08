def problem_p02399():
    (a, b) = [int(x) for x in input().split()]

    x = a // b

    y = a % b

    z = a / b

    print(("{0} {1} {2:.6f}".format(x, y, z)))


problem_p02399()
