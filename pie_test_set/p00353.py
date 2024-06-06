def problem_p00353(input_data):
    m, f, b = list(map(int, input_data.split()))

    return "NA" if m + f < b else max(0, b - m)
