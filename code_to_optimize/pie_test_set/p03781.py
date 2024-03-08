def problem_p03781():
    X = int(eval(input()))

    for n in range(1000000):

        # print((n * (n - 1) // 2))

        if (n * (n - 1) // 2) >= X:

            break

    print((n - 1))


problem_p03781()
