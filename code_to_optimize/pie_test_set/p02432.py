def problem_p02432():
    # -*- coding: utf-8 -*-

    """

    Created on Sat May 19 15:10:55 2018

    AOJ ITP2_1_B

    @author: maezawa

    """

    # import numpy as np

    q = int(eval(input()))

    a = [0 for _ in range(8000000)]

    start = 4000000

    end = 4000001

    for i in range(q):

        ops = list(map(int, input().split()))

        if ops[0] == 0:

            if ops[1] == 0:

                a[start] = ops[2]

                start -= 1

            else:

                a[end] = ops[2]

                end += 1

        elif ops[0] == 1:

            print((a[start + ops[1] + 1]))

        elif ops[0] == 2:

            if ops[1] == 0:

                start += 1

            elif ops[1] == 1:

                end -= 1

    #    print(a[start+1:end])


problem_p02432()
