def problem_p02570(input_data):
    D, T, S = list(map(int, input_data.split()))

    a = T * S - D

    if a >= 0:

        return "Yes"

    else:

        return "No"
