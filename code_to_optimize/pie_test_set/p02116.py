def problem_p02116(input_data):
    n = int(eval(input_data))

    return (n + 1) & -(n + 1)
