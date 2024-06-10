def problem_p02811(input_data):
    n, k = list(map(int, input_data.split()))

    g = 500 * n

    if g >= k:

        return "Yes"

    else:

        return "No"
