def problem_p03543(input_data):
    n = list(map(int, eval(input_data)))

    if n[0] == n[1] == n[2] or n[1] == n[2] == n[3]:

        return "Yes"

    else:

        return "No"
