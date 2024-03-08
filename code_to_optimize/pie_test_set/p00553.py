def problem_p00553():
    a, b, c, d, e = [int(eval(input())) for _ in range(5)]

    if a < 0:

        print((-a * c + d + b * e))

    else:

        print(((b - a) * e))


problem_p00553()
