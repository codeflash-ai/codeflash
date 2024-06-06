def problem_p00018(input_data):
    return " ".join(map(str, sorted(map(int, input_data.split())))[::-1])
