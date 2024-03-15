def problem_p03079(input_data):
    return "YNeos"[len(set(input_data.split())) > 1 :: 2]
