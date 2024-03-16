def problem_p03042(input_data):
    S = eval(input_data)

    a = S[:2]

    b = S[2:]

    if 1 <= int(a) <= 12 and 1 <= int(b) <= 12:

        return "AMBIGUOUS"

    elif 0 <= int(a) <= 99 and 1 <= int(b) <= 12:

        return "YYMM"

    elif 1 <= int(a) <= 12 and 0 <= int(b) <= 99:

        return "MMYY"

    else:

        return "NA"
