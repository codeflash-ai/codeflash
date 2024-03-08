def problem_p00156():
    def ReadMap(n, m):

        Map = [0] * m

        for i in range(m):

            a = input()

            b = list(a)

            if a.count("&") > 0:

                j = a.index("&")

                PosCas = [j, i]

                b[j] = "."

            Map[i] = b

        return Map, PosCas

    def fill(SP, c1, c2):

        for x, y in SP:

            if not (0 <= x < n and 0 <= y < m):

                continue

            if Map[y][x] != c1:

                continue

            if x in [0, n - 1] or y in [0, m - 1]:

                return 1

            Map[y][x] = c2

            SP += [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]

        return 0

    def PrintMap():

        for e in Map:

            print("".join(e))

        print()

        return

    while 1:

        n, m = list(map(int, input().split()))

        if n == m == 0:
            break

        Map, PosCas = ReadMap(n, m)

        c = 0

        #  PrintMap()

        while 1:

            SP = [PosCas]

            if fill(SP, ".", "#"):
                break

            #    PrintMap()

            c += 1

            if fill(SP, "#", "."):
                break

        #    PrintMap()

        print(c)


problem_p00156()
