def problem_p02729(input_data):
    N, M = list(map(int, input_data.split()))

    if N == 1 and M == 1:

        return 0

    elif N == 1 and M != 1:

        return int(M * (M - 1) / 2)

    elif N != 1 and M == 1:

        return int(N * (N - 1) / 2)

    else:

        a = N * (N - 1)

        b = M * (M - 1)

        return int((a + b) / 2)
