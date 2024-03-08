def problem_p03817():
    N = int(eval(input()))

    if N < 11:

        d, m = divmod(N, 6)

        print((d + (m > 0)))

    else:

        d, m = divmod(N, 11)

        a = (1 if m else 0) if m < 7 else 2

        print((d * 2 + a))


problem_p03817()
