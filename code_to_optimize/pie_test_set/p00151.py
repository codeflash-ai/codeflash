def problem_p00151():
    def main():

        while True:

            n = int(eval(input()))

            if n == 0:

                break

            mp = ["0" + eval(input()) + "0" for _ in range(n)]

            mp.insert(0, "0" * (n + 2))

            mp.append("0" * (n + 2))

            score = [[[0] * 4 for _ in range(n + 2)] for _ in range(n + 2)]

            max_score = 0

            for i in range(1, n + 1):

                for j in range(1, n + 1):

                    if mp[i][j] == "1":

                        score[i][j][0] = score[i - 1][j][0] + 1

                        score[i][j][1] = score[i][j - 1][1] + 1

                        score[i][j][2] = score[i - 1][j - 1][2] + 1

                        score[i][j][3] = score[i - 1][j + 1][3] + 1

                        max_score = max(
                            max_score,
                            score[i][j][0],
                            score[i][j][1],
                            score[i][j][2],
                            score[i][j][3],
                        )

                    else:

                        for k in range(4):

                            score[i][j][k] = 0

            print(max_score)

    main()


problem_p00151()
