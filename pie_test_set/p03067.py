def problem_p03067(input_data):
    a, b, c = list(map(int, input_data.split()))

    if a < c < b or b < c < a:

        return "Yes"

    else:

        return "No"
