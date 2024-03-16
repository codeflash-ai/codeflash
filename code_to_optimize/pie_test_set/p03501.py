def problem_p03501(input_data):
    N, A, B = (int(x) for x in input_data.split())

    return min(A * N, B)
