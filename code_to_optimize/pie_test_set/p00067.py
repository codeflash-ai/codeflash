def problem_p00067():
    def island(f, x, y, m):

        rf = f[:][:]

        rf[y][x] = m

        for i in [-1, 1]:

            if 0 <= x + i <= 11 and rf[y][x + i] == 1:

                rf = island(rf, x + i, y, m)

            if 0 <= y + i <= 11 and rf[y + i][x] == 1:

                rf = island(rf, x, y + i, m)

        return rf

    while True:

        f = []
        m = 2

        for i in range(12):

            f.append(list(map(int, list(input()))))

        for y in range(12):

            for x in range(12):

                if f[y][x] == 1:

                    f = island(f, x, y, m)

                    m += 1

        ans = []

        for i in range(12):

            for j in range(12):

                ans.append(f[i][j])

        ans = list(set(ans))

        print(len(ans) - 1 * ans.count(0))

        try:
            input()

        except:
            break


problem_p00067()
