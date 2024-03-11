def problem_p03416(input_data):
    counter = 0

    c = list(map(int, input_data.split(" ")))

    for i in range(c[0], c[1] + 1):

        if str(i)[0] != str(i)[4]:

            continue

        if str(i)[1] != str(i)[3]:

            continue

        counter += 1

    return counter
