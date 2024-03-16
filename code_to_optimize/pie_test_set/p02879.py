def problem_p02879(input_data):
    A, B = list(map(int, input_data.split()))

    if 1 <= A <= 9 and 1 <= B <= 9:

        return A * B

    else:

        return -1
