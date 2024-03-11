def problem_p02719(input_data):
    N, K = list(map(int, input_data.split()))

    r = N % K

    return min(r, K - r)
