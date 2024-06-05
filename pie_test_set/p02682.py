def problem_p02682(input_data):
    A, B, C, K = list(map(int, input_data.split()))

    if A >= K:

        return K

    elif A + B >= K:

        return A

    else:

        return A - (K - (A + B))
