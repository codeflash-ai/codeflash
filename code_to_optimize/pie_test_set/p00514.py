def problem_p00514(input_data):
    n, m, r = list(map(int, input_data.split()))

    a, b = sorted([n - 1, r - m * n])

    if a < 1:
        return 0

    else:

        d = 1

        for i in range(a):
            d *= i + 1

        u = 1

        for i in range(b, a + b):
            u *= i + 1

        return u // d
