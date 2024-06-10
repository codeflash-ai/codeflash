def problem_p03186(input_data):
    A, B, C = list(map(int, input_data.split()))

    return min(C, A + B + 1) + B
