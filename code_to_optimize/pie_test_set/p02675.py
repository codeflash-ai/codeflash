def problem_p02675(input_data):
    n = eval(input_data)

    if n[-1] == "3":

        return "bon"

    elif n[-1] in ["0", "1", "6", "8"]:

        return "pon"

    else:

        return "hon"
