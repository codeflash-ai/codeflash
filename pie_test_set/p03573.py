def problem_p03573(input_data):
    a, b, c = list(map(int, input_data.split()))

    if a == b:

        return c

    elif a == c:

        return b

    else:

        return a
