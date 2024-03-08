def problem_p00139():
    for i in [0] * eval(input()):

        s = input()

        n = (len(s) - 4) / 2

        if s == ">'" + "=" * n + "#" + "=" * n + "~":
            t = "A"

        elif s == ">^" + "Q=" * n + "~~":
            t = "B"

        else:
            t = "NA"

        if n < 1:
            t = "NA"

        print(t)


problem_p00139()
