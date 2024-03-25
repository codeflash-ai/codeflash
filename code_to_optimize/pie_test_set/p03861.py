def problem_p03861(input_data):
    import math

    a, b, x = list(map(int, input_data.split()))

    return b // x - (a - 1) // x
