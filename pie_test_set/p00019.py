def problem_p00019(input_data):
    from functools import reduce as R

    return R(lambda x, y: x * y, list(range(1, int(eval(input_data)) + 1)))
