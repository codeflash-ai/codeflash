def problem_p00103():
    n = eval(input())

    for _ in [0] * n:

        i = 0

        b = 0

        c = 0

        while i < 3:

            s = input()[1]

            if s == "I":

                if b == 3:
                    c += 1

                else:
                    b += 1

            elif s == "O":

                c += b + 1

                b = 0

            elif s == "U":
                i += 1

        print(c)


problem_p00103()
