def problem_p02818(input_data):
    a, b, k = list(map(int, input_data.split()))

    if a + b <= k:

        return (0, 0)

    else:

        return (0 if a < k else a - k, b + (a - k) if a < k else b)
