def problem_p02885(input_data):
    a, b = list(map(int, input_data.split()))

    c = a - b * 2

    if c < 0:

        return 0

    else:

        return c
