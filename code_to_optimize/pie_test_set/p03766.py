def problem_p03766(input_data):
    n = int(eval(input_data))
    a, b, c, p = 1, 1, n, n - 1

    for i in range(n - 1):

        p += a - 1
        a, b, c = b, c, ((n - 1) ** 2 + p + c) % (10**9 + 7)

    return c
