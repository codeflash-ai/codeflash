def problem_p02790(input_data):
    a, b = input_data.split()

    return a * int(b) if a < b else b * int(a)
