def problem_p02981(input_data):
    N, A, B = list(map(int, input_data.split()))

    return min(N * A, B)
