def problem_p03574():
    import numpy as np

    h, w = input().split()

    h, w = int(h), int(w)

    t1 = []

    for i in range(h):

        s1 = list(input())

        t1.append(s1)

    t2 = [[0 for i in range(w)] for i in range(h)]

    for i in range(h):

        for j in range(w):

            if t1[i][j] == "#":

                for k in range(3):

                    for l in range(3):

                        if k == 1 and l == 1:

                            continue

                        if (
                            i + k - 1 >= 0
                            and j + l - 1 >= 0
                            and i + k - 1 < h
                            and j + l - 1 < w
                            and t1[i + k - 1][j + l - 1] != "#"
                        ):

                            t2[i + k - 1][j + l - 1] += 1

                t2[i][j] = "#"

    for i in range(h):

        for j in range(w):

            print(t2[i][j], end="")

        print("")


problem_p03574()
