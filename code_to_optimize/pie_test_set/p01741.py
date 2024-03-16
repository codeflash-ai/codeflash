def problem_p01741(input_data):
    import math

    n = float(eval(input_data))

    return int(n) + 1 if n * math.sqrt(2) < int(n) + 1 else n * math.sqrt(2)
