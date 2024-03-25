def problem_p03815(input_data):
    n = int(input_data)

    c = 2 * (n / 11)

    h = {0: 0}

    for i in range(1, 6 + 1):

        h[i] = 1

    for i in range(7, 12):

        h[i] = 2

    if n in h:

        return h[n]

    else:

        return h[n % 11] + c
