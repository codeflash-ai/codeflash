def problem_p03763():
    n = int(input())

    ss = "abcdefghijklmnopqrstuvwxyz"

    dmin = {}

    for s in ss:

        dmin[s] = 51

    for i in range(n):

        S = input()

        d = {}

        for s in S:

            if s not in d.keys():

                d[s] = 1

            else:

                d[s] += 1

        for s in ss:

            if s not in d.keys():

                dmin[s] = 0

        for s in d.keys():

            dmin[s] = min(dmin[s], d[s])

    for k in dmin.keys():

        print(k * dmin[k], sep="", end="")

    print()


problem_p03763()
