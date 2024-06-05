def problem_p03371(input_data):
    A, B, C, X, Y = list(map(int, input_data.split()))

    ans = 5000 * 2 * 10**5

    for i in range(10**5 + 1):

        s = i * 2 * C + max(0, X - i) * A + max(0, Y - i) * B

        ans = min(ans, s)

    return ans
