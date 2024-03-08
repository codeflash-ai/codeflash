def problem_p00030():
    def dec2bin(n, base=10):

        s = format(n, "b")

        s = "0" * (10 - len(s)) + s

        return s

    x = [[0] * 50 for i in range(11)]

    for i in range(0, 2**10):

        B = dec2bin(i)

        s = 0

        c = B.count("1")

        j = 0

        for e in B:

            if e == "1":
                s += j

            j += 1

        x[c][s] += 1

    while 1:

        n, s = list(map(int, input().split()))

        if n == 0 and s == 0:
            break

        if s >= 50:
            print(0)

        else:
            print(x[n][s])


problem_p00030()
