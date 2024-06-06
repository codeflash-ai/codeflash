def problem_p03863(input_data):
    S = eval(input_data)

    if S[0] == S[-1]:

        return "Second" if len(S) % 2 else "First"

    else:

        return "Second" if len(S) % 2 == 0 else "First"
