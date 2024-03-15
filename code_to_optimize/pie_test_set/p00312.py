def problem_p00312(input_data):
    D, L = list(map(int, input_data.split()))

    return sum(divmod(D, L))
