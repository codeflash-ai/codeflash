def problem_p03227(input_data):
    s = eval(input_data)

    if len(s) < 3:
        return s

    else:
        return s[::-1]
