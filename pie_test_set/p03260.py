def problem_p03260(input_data):
    a, b = list(map(int, input_data.split()))

    return "Yes" if (a * b) % 2 else "No"
