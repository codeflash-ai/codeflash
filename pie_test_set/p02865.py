def problem_p02865(input_data):
    n = int(eval(input_data))

    if n % 2 == 0:

        return n // 2 - 1

    else:

        return (n + 1) // 2 - 1
