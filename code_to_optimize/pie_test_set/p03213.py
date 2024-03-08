def problem_p03213():
    from operator import mul

    from functools import reduce

    nCr = {}

    def cmb(n, r):

        if r == 0 or r == n:
            return 1

        if r == 1:
            return n

        if (n, r) in nCr:
            return nCr[(n, r)]

        nCr[(n, r)] = cmb(n - 1, r) + cmb(n - 1, r - 1)

        return nCr[(n, r)]

    a = int(eval(input()))

    b = [0 for i in range(15)]

    j = 0

    c = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    d = [0 for i in range(101)]

    p = 0

    pp = 0

    g = 0

    gg = 0

    ggg = 0

    co = 0

    for i in range(1, 101):

        k = i

        while True:

            if k % c[j] == 0 and k >= c[j]:

                b[j] += 1

                k //= c[j]

            else:

                j += 1

                if j == 15:

                    j = 0

                    break

        for k in range(15):

            if b[k] >= 74:

                g += 1

            if b[k] >= 24:

                gg += 1

            if b[k] >= 14:

                ggg += 1

            if b[k] >= 4:

                p += 1

            if b[k] >= 2:

                pp += 1

        if i >= 10:

            d[i] = (
                cmb(p, 2) * cmb((pp - 2), 1)
                + cmb(g, 1)
                + cmb(gg, 1) * cmb((pp - 1), 1)
                + cmb(ggg, 1) * cmb((p - 1), 1)
            )

        p = 0

        pp = 0

        g = 0

        gg = 0

        ggg = 0

    print((d[a]))


problem_p03213()
