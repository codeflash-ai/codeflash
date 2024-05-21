def problem_p03963(input_data):
    N, K = [int(i) for i in input_data.split()]

    ans = K

    for _ in range(N - 1):

        ans *= K - 1

    return ans
