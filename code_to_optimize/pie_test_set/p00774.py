def problem_p00774():
    while 1:

        H = int(input())

        if H == 0:

            break

        Board = [[0 for j in range(H + 1)] for i in range(6)]

        for j in range(H):

            inputs = list(map(int, input().split(" ")))

            for i in range(5):

                Board[i][H - j] = inputs[i]

        total_score = 0

        tmp_score = -1

        while tmp_score != 0:

            tmp_score = 0

            tmpBoard = [[0] for i in range(6)]

            for j in range(1, H + 1):

                con_num = 1

                for i in range(1, 6):

                    if Board[i][j] == Board[i - 1][j]:

                        con_num += 1

                    else:

                        if con_num >= 3:

                            tmp_score += con_num * Board[i - 1][j]

                        else:

                            for h in range(con_num):

                                tmpBoard[i - 1 - h].append(Board[i - 1 - h][j])

                                tmpBoard[i - 1 - h][0] += 1

                        con_num = 1

            total_score += tmp_score

            Board = [line + [0 for j in range(H - line[0])] for line in tmpBoard]

        print(total_score)


problem_p00774()
