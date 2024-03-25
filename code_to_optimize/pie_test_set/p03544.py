def problem_p03544(input_data):
    N = int(eval(input_data))

    L = [0] * 1000000

    L[0] = 2

    L[1] = 1

    for i in range(2, N + 1):

        L[i] = L[i - 2] + L[i - 1]

    return L[N]
