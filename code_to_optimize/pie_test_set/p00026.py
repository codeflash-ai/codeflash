def problem_p00026():
    drops = {
        1: ((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 1, 1, 1, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),
        2: ((0, 0, 0, 0, 0), (0, 1, 1, 1, 0), (0, 1, 1, 1, 0), (0, 1, 1, 1, 0), (0, 0, 0, 0, 0)),
        3: ((0, 0, 1, 0, 0), (0, 1, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 1, 1, 0), (0, 0, 1, 0, 0)),
    }

    paper = [[0 for i in range(14)] for j in range(14)]

    def solve(x, y, size):

        for i in range(5):

            for j in range(5):

                if drops[size][i][j] == 1:

                    paper[y + i][x + j] += 1

    while True:

        try:

            line = input()

        except EOFError:

            break

        x, y, size = list(map(int, line.split(",")))

        solve(x, y, size)

    s = c = 0

    for i in range(2, 12):

        for j in range(2, 12):

            if paper[i][j] == 0:

                c += 1

            if s < paper[i][j]:

                s = paper[i][j]

    print(c)

    print(s)


problem_p00026()
