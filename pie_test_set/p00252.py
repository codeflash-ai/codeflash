def problem_p00252(input_data):
    a, b, c = list(map(int, input_data.split()))

    return ["Close", "Open"][(a & b) ^ c]
