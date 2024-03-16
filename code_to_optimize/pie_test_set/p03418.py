def problem_p03418(input_data):
    N, K = list(map(int, input_data.split()))

    if K == 0:

        return N * N

        exit()

    ans = 0

    for b in range(K + 1, N + 1):

        p = N // b

        ans += p * max(0, b - K) + max(0, N - p * b - K + 1)

    return ans
