def problem_p03005(input_data):
    N, K = list(map(int, input_data.split()))

    if K > 1:

        return N - K

    else:

        return 0
