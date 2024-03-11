def problem_p03289(input_data):
    import string

    li = list(string.ascii_lowercase)

    s = list(eval(input_data))

    if s[0] == "A" and s[2:-1].count("C") == 1:

        s.remove("C")

        s.remove("A")

    else:

        return "WA"

        exit()

    if all([s[i] in li for i in range(len(s))]):

        return "AC"

        exit()

    else:

        return "WA"
