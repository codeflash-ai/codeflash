def problem_p03779():
    X = int(eval(input()))

    i, x = 1, 0

    while x < X:

        x += i

        i += 1

    print((i - 1))


problem_p03779()
