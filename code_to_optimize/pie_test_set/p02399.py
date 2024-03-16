def problem_p02399(input_data):
    (a, b) = [int(x) for x in input_data.split()]

    x = a // b

    y = a % b

    z = a / b

    return "{0} {1} {2:.6f}".format(x, y, z)
