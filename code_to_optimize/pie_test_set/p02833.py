def problem_p02833(input_data):
    N = int(eval(input_data))

    if N % 2 == 1:

        return 0

        exit()

    ans = 0

    mod = 10

    while mod <= N:

        ans += N // mod

        mod *= 5

    return ans
