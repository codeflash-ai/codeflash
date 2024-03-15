def problem_p03265(input_data):
    x1, y1, x2, y2 = list(map(int, input_data.split()))

    x3 = x2 - (y2 - y1)

    y3 = y2 + (x2 - x1)

    x4 = x3 - (y3 - y2)

    y4 = y3 + (x3 - x2)

    return (x3, y3, x4, y4)
