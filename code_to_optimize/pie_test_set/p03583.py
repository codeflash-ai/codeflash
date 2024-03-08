def problem_p03583():
    N = int(eval(input()))

    for a in range(1, 3501):

        for b in range(1, 3501):

            m = (4 * a * b) - N * (a + b)

            if m > 0 and N * a * b % m == 0:

                c = (N * a * b) // m

                break

        else:

            continue

        break

    print((a, b, c))


problem_p03583()
