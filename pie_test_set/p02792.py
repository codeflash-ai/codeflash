def problem_p02792(input_data):
    N = int(eval(input_data))

    C = [[0] * 9 for i in range(9)]

    count = 0

    for n in range(1, N + 1):

        if str(n)[-1] != "0":

            left = int(str(n)[0]) - 1

            right = int(str(n)[-1]) - 1

            C[left][right] += 1

    for i in range(1, 10):

        for j in range(1, 10):

            count += C[i - 1][j - 1] * C[j - 1][i - 1]

    return count
