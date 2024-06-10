def problem_p02400(input_data):
    import math

    r = float(eval(input_data))

    a = math.pi * r**2

    b = math.pi * r * 2

    return str(a) + " " + str(b)
