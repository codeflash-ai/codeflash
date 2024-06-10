def problem_p03705(input_data):
    N, A, B = list(map(int, input_data.split()))

    if N == 1:

        return 1 if A == B else 0

    else:

        if A > B:

            return 0

        else:

            m = A * (N - 1) + B

            M = B * (N - 1) + A

            return M - m + 1
