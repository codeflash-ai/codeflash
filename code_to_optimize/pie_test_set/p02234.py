def problem_p02234():
    def s():

        n = int(eval(input())) + 1

        e = [input().split() for _ in [0] * ~-n]

        p = [int(e[0][0])] + list(int(x[1]) for x in e)

        m = [[0] * n for _ in [0] * n]

        for l in range(2, n):

            for i in range(1, n - l + 1):

                j = i + l - 1
                m[i][j] = 1e6

                for k in range(i, j):
                    m[i][j] = min(m[i][j], m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j])

        print((m[1][n - 1]))

    if "__main__" == __name__:
        s()


problem_p02234()
