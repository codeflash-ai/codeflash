def problem_p03288(input_data):
    r = int(eval(input_data))

    if r < 1200:

        return "ABC"

    elif r < 2800:

        return "ARC"

    else:

        return "AGC"
