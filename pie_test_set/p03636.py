def problem_p03636(input_data):
    A = eval(input_data)

    return A[0] + str(len(A[1:-1])) + A[-1]
