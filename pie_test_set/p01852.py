def problem_p01852(input_data):
    n = int(eval(input_data))

    return 0 if n == 0 else len(str(bin(n))[2:])
