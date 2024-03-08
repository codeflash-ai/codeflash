def problem_p00131():
    def f(x, y):

        global B

        B[y][x] ^= 1

        if y > 0:
            B[y - 1][x] ^= 1

        if y < 9:
            B[y + 1][x] ^= 1

        if x > 0:
            B[y][x - 1] ^= 1

        if x < 9:
            B[y][x + 1] ^= 1

        return (x, y)

    R = list(range(10))

    n = eval(input())

    for _ in [0] * n:

        A = [list(map(int, input().split())) for _ in R]

        for p in range(1024):

            B = [e[:] for e in A]

            x = []

            for j in R:

                a = [list(map(int, list(format(p, "010b")))), B[j - 1]][j != 0]

                for i in R:

                    if a[i] == 1:
                        x.append(f(i, j))

            if sum(B[9]) == 0:
                break

        for i, j in x:
            B[j][i] = 1

        for e in B:
            print(" ".join(map(str, e)))


problem_p00131()
