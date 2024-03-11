def problem_p03139(input_data):
    N, A, B = list(map(int, input_data.split()))

    if A + B - N > 0:

        return (min(A, B), A + B - N)

    else:

        return (min(A, B), "0")
