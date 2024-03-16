def problem_p02546(input_data):
    S = eval(input_data)

    if S.endswith("s"):

        S = S + "es"

    else:

        S = S + "s"

    return S
