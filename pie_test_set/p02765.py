def problem_p02765(input_data):
    N, R = list(map(int, input_data.split()))

    return R + max(0, 10 - N) * 100
