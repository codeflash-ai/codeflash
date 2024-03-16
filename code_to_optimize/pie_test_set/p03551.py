def problem_p03551(input_data):
    N, M = list(map(int, input_data.split()))

    total_time = M * 1900 + (N - M) * 100

    e = 0

    err = 10 ** (-3)

    prev = -100

    i = 0

    while abs(prev - e) > err:

        prev = e

        e += (1 - (1 / 2) ** M) ** i * (1 / 2) ** M * total_time * (i + 1)

        i += 1

    def floor(x, y):

        return ((-x) // y) * (-1)

    return int(floor(e, 1))
