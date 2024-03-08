def problem_p03146():
    a = [0] * 1000000

    a[0] = int(eval(input()))

    for i in range(1, 1000000):

        if a[i - 1] % 2 == 0:

            if a[i - 1] / 2 in a:

                print((i + 1))

                break

            a[i] = a[i - 1] / 2

        else:

            if 3 * a[i - 1] + 1 in a:

                print((i + 1))

                break

            a[i] = 3 * a[i - 1] + 1


problem_p03146()
