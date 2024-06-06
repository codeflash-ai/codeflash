def problem_p03328(input_data):
    a, b = (int(i) for i in input_data.split())

    return sum(range(b - a)) - a
