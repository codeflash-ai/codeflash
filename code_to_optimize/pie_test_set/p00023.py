def problem_p00023():
    # -*- coding: utf-8 -*-

    import sys

    import os

    import math

    N = int(eval(input()))

    for i in range(N):

        ax, ay, ar, bx, by, br = list(map(float, input().split()))

        between_center = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

        # ????????£????????????

        if between_center > ar + br:

            print((0))

        # ????????????????????¨

        else:

            # B in A

            if ar > between_center + br:

                print((2))

            # A in B

            elif br > between_center + ar:

                print((-2))

            else:

                print((1))


problem_p00023()
