def problem_p02759(input_data):
    n = int(eval(input_data))

    return int(n / 2) if n % 2 == 0 else n // 2 + 1
