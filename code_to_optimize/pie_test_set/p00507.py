def problem_p00507():
    def con(m, n):

        ln = len(str(n))

        return m * 10**ln + n

    n = int(eval(input()))

    lst = [int(eval(input())) for i in range(n)]

    lst.sort()

    lst = lst[0:4]

    save = []

    for i in lst:

        for j in lst:

            if i != j:

                save.append(con(i, j))

    save.sort()

    print((save[2]))


problem_p00507()
