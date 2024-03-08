def problem_p03696():
    import numpy as np

    n = int(eval(input()))

    s = list(eval(input()))

    j = []

    now = 0

    unclosen = 0

    opened = 0

    for i in s:

        if i == ")":

            now -= 1

            if opened:

                opened -= 1

            else:

                unclosen += 1

        else:

            now += 1

            opened += 1

    j = np.array(j)

    print(("(" * unclosen + "".join(s) + ")" * opened))


problem_p03696()
