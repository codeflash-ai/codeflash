def problem_p03419(input_data):
    N, M = list(map(int, input_data.split()))

    if N == 2 or M == 2:

        return 0

    else:

        return abs((N - 2) * (M - 2))
