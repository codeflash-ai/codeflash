def problem_p02659(input_data):
    a, b = input_data.split()
    return int(a) * int(b[:-3] + b[-2:]) // 100
