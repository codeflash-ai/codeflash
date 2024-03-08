def problem_p00102():
    while True:

        n = int(input())

        if n == 0:

            break

        row = n

        tab = []

        for i in range(row):

            r = list(map(int, input().split()))

            r.append(sum(r))

            tab.append(r)

        bel = []

        for i in range(row):

            tmp = 0

            for j in range(row):

                tmp += tab[j][i]

            bel.append(tmp)

        bel.append(sum(bel))

        tab.append(bel)

        for i in range(row + 1):

            for j in range(row + 1):

                # print("".join(map("{0:>5}".format, str(tab[i][j]))))

                print("{:>5}".format(str(tab[i][j])), end="")

            print()


problem_p00102()
