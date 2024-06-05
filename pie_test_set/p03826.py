def problem_p03826(input_data):
    a, b, c, d = list(map(int, input_data.split()))

    return int(max(a * b, c * d))
