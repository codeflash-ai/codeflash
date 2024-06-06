def problem_p03359(input_data):
    a, b = list(map(int, input_data.split()))
    return a - (a > b)
