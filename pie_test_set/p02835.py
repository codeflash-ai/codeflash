def problem_p02835(input_data):
    a, b, c = list(map(int, input_data.split()))

    return "bust" if a + b + c >= 22 else "win"
