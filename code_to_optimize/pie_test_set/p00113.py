def problem_p00113():
    from decimal import *

    import re

    def solve2(m, n):

        maxlen = 160

        PREC = 200

        getcontext().prec = PREC

        x = Decimal(m) / Decimal(n)

        s = x.to_eng_string()

        if len(s) < PREC:

            return (s[2:], "")

        rep = 1

        while True:

            r = r"(.{%d})\1{%d,}" % (rep, int(maxlen / rep) - 1)  # ex. '(.{6})\\1{12,}'

            a = re.search(r, s)

            if a:

                break

            rep += 1

            if rep > maxlen:

                raise ValueError("This cannot happen.rep=%d" % rep)

        u = s[2 : a.start() + len(a.group(1))]

        v = (" " * PREC + "^" * len(a.group(1)))[-len(u) :]

        return (u, v)

    while True:

        try:

            m, n = list(map(int, input().strip().split()))

            s, t = solve2(m, n)

            print(s)

            if t != "":

                print(t)

        except EOFError:

            break


problem_p00113()
