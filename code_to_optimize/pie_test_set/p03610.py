def problem_p03610(input_data):
    s = eval(input_data)

    odd_string = ""

    for i, c in enumerate(s):

        if i % 2 == 0:

            odd_string += c

    return odd_string
