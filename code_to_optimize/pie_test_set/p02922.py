def problem_p02922(input_data):
    A, B = list(map(int, input_data.split()))

    start = 1

    for i in range(20):

        if start >= B:

            return i

            break

        start += A - 1
