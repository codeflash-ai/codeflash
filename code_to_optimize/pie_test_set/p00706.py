def problem_p00706():
    while 1:

        n = int(input())

        if n == 0:

            break

        w, h = list(map(int, input().split(" ")))

        field = [[0 for i in range(0, w + 1)] for j in range(0, h + 1)]

        for i in range(n):

            x, y = list(map(int, input().split(" ")))

            field[y][x] = 1

        s, t = list(map(int, input().split(" ")))

        tree_max = 0

        for i in range(1, w - s + 2):

            for j in range(1, h - t + 2):

                tmp_sum = 0

                for k in range(i, i + s):

                    for l in range(j, j + t):

                        tmp_sum += field[l][k]

                tree_max = max(tree_max, tmp_sum)

        print(tree_max)


problem_p00706()
