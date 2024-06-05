def problem_p03455(input_data):
    a, b = list(map(int, input_data.split()))

    if a * b % 2 == 1:

        return "Odd"

    else:

        return "Even"
