def problem_p03548(input_data):
    X, Y, Z = list(map(int, input_data.split()))

    for n in range(10**5, 0, -1):

        if X >= n * (Y + Z) + Z:

            break

    return n
