def problem_p03035(input_data):
    a, b = list(map(int, input_data.split()))

    return b if 13 <= a else b // 2 if 6 <= a else 0
