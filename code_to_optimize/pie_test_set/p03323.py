def problem_p03323(input_data):
    a, b = list(map(int, input_data.split()))

    if a >= 9 or b >= 9:
        return ":("

    else:
        return "Yay!"
