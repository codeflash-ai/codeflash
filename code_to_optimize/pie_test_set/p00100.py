def problem_p00100():
    while 1:

        n = int(eval(input()))

        if not n:
            break

        f = 1

        L = {}

        for a, b, c in [list(map(int, input().split())) for i in range(n)]:

            a = str(a)

            d = b * c

            if a in L:

                if L[a] >= 1e6:
                    continue

                L[a] += d

            else:
                L[a] = d

            if L[a] >= 1e6:

                print(a)

                f = 0

        if f:
            print("NA")


problem_p00100()
