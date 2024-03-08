def problem_p02950():
    p = eval(input())

    a = list(map(int, input().strip().split()))

    b = [0] * p

    p2 = p * p

    s = []

    for i in range(p):

        if a[i]:
            s.append(i)

    ncr = [[0 for i in range(p)] for j in range(p)]

    for i in range(p):

        ncr[i][0] = ncr[i][i] = 1

        ncr[i][1] = i

    for n in range(2, p):

        for r in range(2, n):

            ncr[n][r] = ncr[n - 1][r] + ncr[n - 1][r - 1]

            if ncr[n][r] >= p:
                ncr[n][r] -= p

    jpow = [[0 for i in range(p)] for j in range(p)]

    jpow[0][0] = 1

    for i in range(1, p):

        val = 1

        for j in range(p):

            jpow[i][j] = val

            val *= i

            if val >= p2:
                val %= p

    for r in range(1, p):

        for j in s:

            x = ncr[p - 1][r]

            x *= jpow[j][p - 1 - r]

            if x >= p2:
                x %= p

            if (p - r) & 1:
                b[r] -= x

            else:
                b[r] += x

        if b[r] >= p or b[r] < 0:
            b[r] %= p

    if a[0]:
        b[0] = 1

    print(" ".join(map(str, b)))


problem_p02950()
