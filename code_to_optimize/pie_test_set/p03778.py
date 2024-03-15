def problem_p03778(input_data):
    W, a, b = list(map(int, input_data.split()))

    s = {i for i in range(a, a + W + 1)} & {i for i in range(b, b + W + 1)}

    if s:

        return 0

    else:

        return min(abs(b - a - W), abs(b + W - a))
