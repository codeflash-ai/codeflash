def problem_p03407(input_data):
    a, b, c = list(map(int, input_data.split()))

    if a + b < c:

        return "No"

    else:

        return "Yes"
