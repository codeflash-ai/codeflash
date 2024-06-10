def problem_p03693(input_data):
    r, g, b = list(map(int, input_data.split()))

    if (10 * g + b) % 4 == 0:

        return "YES"

    else:

        return "NO"
