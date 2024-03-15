def problem_p03592(input_data):
    N, M, K = list(map(int, input_data.split()))

    for i in range(N + 1):

        for j in range(M + 1):

            t = i * M + j * N - i * j * 2

            if t == K:

                return "Yes"

                exit()

    return "No"
