def problem_p00263():
    for i in range(eval(input())):

        b = format(int(input(), 16), "b").zfill(32)

        a1 = 0

        a2 = 0.0

        for i in range(1, 25):

            a1 += int(b[i]) * 2 ** (24 - i)

        for i in range(25, 32):

            a2 += int(b[i]) * 2 ** (24 - i)

        print("-" * int(b[0]) + str(a1) + str(a2)[1:])


problem_p00263()
