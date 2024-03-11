def problem_p03437(input_data):
    X, Y = list(map(int, input_data.split()))

    for i in range(1, 10**6):

        if (X * i) % Y != 0:

            return X * i

            exit()

    return -1
