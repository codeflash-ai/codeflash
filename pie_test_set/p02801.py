def problem_p02801(input_data):
    from string import ascii_lowercase as lower

    return lower[lower.find(eval(input_data)) + 1]
