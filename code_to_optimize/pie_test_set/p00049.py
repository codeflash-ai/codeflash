def problem_p00049():
    import sys

    L = ["A", "B", "AB", "O"]

    b = {}

    for e in L:

        b[e] = 0

    for s in sys.stdin:

        i = s[s.index(",") + 1 : -1]

        b[i] += 1

    for e in L:

        print(b[e])


problem_p00049()
