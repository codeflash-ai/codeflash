def problem_p03781(input_data):
    X = int(eval(input_data))

    for n in range(1000000):

        # return ((n * (n - 1) // 2))

        if (n * (n - 1) // 2) >= X:

            break

    return n - 1
