def problem_p02582(input_data):
    S = eval(input_data)

    if S == "RRR":

        return 3

    elif "RR" in S:

        return 2

    elif "R" in S:

        return 1

    else:

        return 0
