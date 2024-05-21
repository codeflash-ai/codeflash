def problem_p03369(input_data):
    s = eval(input_data)

    count = 0

    if s[0] == "o":

        count += 1

    if s[1] == "o":

        count += 1

    if s[2] == "o":

        count += 1

    return 700 + 100 * count
