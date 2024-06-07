def problem_p02951(input_data):
    a, b, c = list(map(int, input_data.split()))

    if a < b + c:

        return b + c - a

    else:

        return 0
