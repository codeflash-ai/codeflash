def problem_p03796(input_data):
    import math

    n = int(eval(input_data))

    return math.factorial(n) % (10**9 + 7)
