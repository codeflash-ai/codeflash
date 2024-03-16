def problem_p03050(input_data):
    N = int(eval(input_data))

    ans = 0

    for i in range(1, N + 1):

        x = (N - i) // i

        if x <= i:

            break

        if (N - i) % i == 0:

            ans += x

    return ans
