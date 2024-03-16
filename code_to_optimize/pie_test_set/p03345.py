def problem_p03345(input_data):
    A, B, C, K = list(map(int, input_data.split()))

    return B - A if K % 2 else A - B
