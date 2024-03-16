def problem_p03623(input_data):
    x, a, b = list(map(int, input_data.split()))

    A = abs(x - a)

    B = abs(x - b)

    if A < B:

        return "A"

    else:

        return "B"
