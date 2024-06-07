def problem_p03385(input_data):
    s = list(eval(input_data))

    a = s.count("a")

    b = s.count("b")

    c = s.count("c")

    if a == 1 and b == 1 and c == 1:

        return "Yes"

    else:

        return "No"
