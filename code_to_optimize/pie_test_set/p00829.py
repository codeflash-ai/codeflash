def problem_p00829():
    #!/usr/bin/env python

    import sys

    import math

    import itertools as it

    from collections import deque

    sys.setrecursionlimit(10000000)

    N = eval(input())

    for i in range(N):

        lst = []

        while len(lst) < 9:

            lst_ = input().split()

            for S in lst_:

                lst.append(int(S, 16))

        res = lst[8]

        lst = lst[:8]

        diff = (sum(lst) - res) % (1 << 32)

        diff_lst = []

        for i in range(32):

            D = 0

            for j in lst:

                if j & (1 << i) == 0:

                    D += 1 << i

                else:

                    D -= 1 << i

            if res & (1 << i) == 0:

                D -= 1 << i

            else:

                D += 1 << i

            D %= 1 << 32

            diff_lst.append([D, 1 << i])

        L1 = [[diff, 0]]

        for i in range(16):

            P = diff_lst[i]

            L = []

            for p in L1:

                L.append([(p[0] + P[0]) % (1 << 32), p[1] + P[1]])

            L1 += L

        L2 = [[0, 0]]

        for i in range(16, 32):

            P = diff_lst[i]

            L = []

            for p in L2:

                L.append([(p[0] + P[0]) % (1 << 32), p[1] + P[1]])

            L2 += L

        L1.sort()

        L2.sort()

        if L1[0][0] == 0:

            print(hex(L1[0][1])[2:])

            continue

        pos = len(L2) - 1

        for p in L1:

            while p[0] + L2[pos][0] > (1 << 32):

                pos -= 1

            if p[0] + L2[pos][0] == (1 << 32):

                print(hex(p[1] + L2[pos][1])[2:])

                break


problem_p00829()
