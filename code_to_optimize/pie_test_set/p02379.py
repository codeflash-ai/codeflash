def problem_p02379(input_data):
    import math

    x1, y1, x2, y2 = list(map(float, input_data.split(" ")))

    return "{:.5f}".format(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
