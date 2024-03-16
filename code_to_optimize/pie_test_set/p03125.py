def problem_p03125(input_data):
    a, b = list(map(int, input_data.split()))

    if (b % a) == 0:

        return a + b

    else:

        return b - a
