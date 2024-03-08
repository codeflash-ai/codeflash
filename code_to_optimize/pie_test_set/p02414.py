def problem_p02414():
    import sys

    e = [list(map(int, e.split())) for e in sys.stdin]

    n = e[0][0] + 1

    for c in e[1:n]:

        t = ""

        for l in zip(*e[n:]):
            t += f"{sum(s*t for s,t in zip(c,l))} "

        print((t[:-1]))


problem_p02414()
