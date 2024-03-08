def problem_p03503():
    import sys

    import numpy as np

    # N 店の数

    n = int(eval(input()))

    # F 店の営業計画

    f = [input().replace(" ", "") for i in range(n)]

    # P 営業利益

    p = [list(map(int, input().split())) for i in range(n)]

    p_max = -999999999999999

    for i in range(1, 1024):

        f_b = format(i, "010b")

        p_10 = 0

        for f_1, p_1 in zip(f, p):

            p_cnt = 0

            for j in range(10):

                if f_1[j] == "1" and f_b[j] == "1":

                    p_cnt += 1

            p_10 += p_1[p_cnt]

        if p_10 > p_max:

            if p_10 == 0:

                print((str(i)))

                break

            p_max = p_10

    print(p_max)


problem_p03503()
