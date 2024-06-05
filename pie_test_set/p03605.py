def problem_p03605(input_data):
    N = int(eval(input_data))

    if N // 10 == 9 or N % 10 == 9:

        return "Yes"

    else:

        return "No"
