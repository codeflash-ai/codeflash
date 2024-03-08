def problem_p02977():
    n = int(eval(input()))

    b = 1 << (n.bit_length() - 1)

    if n - b == 0:

        print("No")

    else:

        print("Yes")

        print((1, 2))

        print((2, 3))

        print((3, n + 1))

        print((n + 1, n + 2))

        print((n + 2, n + 3))

        for i in range(4, n, 2):

            print((i, i + 1))

            print((i + n, i + n + 1))

            print((1 + n, i + n))

            print((1 + n, i + 1))

        if (~n) & 1:

            print((n, b + n))

            a = n ^ b ^ 1

            print((n + n, a))


problem_p02977()
