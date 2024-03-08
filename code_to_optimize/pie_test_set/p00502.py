def problem_p00502():
    INF = 10**20

    def main():

        d, n = list(map(int, input().split()))

        temp = [int(eval(input())) for i in range(d)]

        temp.insert(0, 0)

        alst = []

        blst = []

        clst = []

        for i in range(n):

            a, b, c = list(map(int, input().split()))

            alst.append(a)

            blst.append(b)

            clst.append(c)

        dp = [[0] * n for i in range(d + 1)]

        t1 = temp[1]

        for i in range(n):

            if not (alst[i] <= t1 <= blst[i]):

                dp[1][i] = -INF

        for i in range(2, d + 1):

            t = temp[i]

            predp = dp[i - 1]

            for j in range(n):

                cj = clst[j]

                if alst[j] <= t <= blst[j]:

                    dp[i][j] = max(
                        predp[x] + (cj - clst[x] if cj >= clst[x] else clst[x] - cj)
                        for x in range(n)
                    )

        print((max(dp[d])))

    main()


problem_p00502()
