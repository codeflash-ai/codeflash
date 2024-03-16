def problem_p03337(input_data):
    A, B = list(map(int, input_data.split()))

    C = [A + B, A - B, A * B]

    return max(C)
