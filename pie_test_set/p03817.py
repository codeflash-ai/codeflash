def problem_p03817(input_data):
    N = int(eval(input_data))

    if N < 11:

        d, m = divmod(N, 6)

        return d + (m > 0)

    else:

        d, m = divmod(N, 11)

        a = (1 if m else 0) if m < 7 else 2

        return d * 2 + a
