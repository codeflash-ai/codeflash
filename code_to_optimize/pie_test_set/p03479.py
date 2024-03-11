def problem_p03479(input_data):
    X, Y = list(map(int, input_data.split()))

    t = X

    ans = 0

    while t <= Y:

        t *= 2

        ans += 1

    return ans
