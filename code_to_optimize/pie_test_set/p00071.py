def problem_p00071():
    def bomb(x, y):

        s = M[y]

        if s[x] == "0":
            return

        M[y] = s[:x] + "0" + s[x + 1 :]

        R = [-3, -2, -1, 1, 2, 3]

        for e in R:

            bomb(x + e, y)

            bomb(x, y + e)

        return

    A = list(range(3, 11))

    M = ["00000000000000" for i in range(14)]

    z = "000"

    n = eval(input())

    for i in range(n):

        s = input()

        for j in A:

            M[j] = z + input() + z

        x = eval(input()) + 2

        y = eval(input()) + 2

        bomb(x, y)

        print("Data %d:" % (i + 1))

        for j in A:

            print(M[j][3:-3])


problem_p00071()
