def problem_p03779(input_data):
    X = int(eval(input_data))

    i, x = 1, 0

    while x < X:

        x += i

        i += 1

    return i - 1
