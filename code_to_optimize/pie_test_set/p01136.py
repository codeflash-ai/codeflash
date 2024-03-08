def problem_p01136():
    MAX_N = 50

    MAX_DAY = 30

    def solve(n, f):

        dp = [[set() for j in range(n)] for i in range(MAX_DAY + 1)]

        for i in range(n):

            dp[0][i].add(i)

        for d in range(1, MAX_DAY + 1):

            # for line in dp[:5]:

            #     print(line)

            for i in range(n):

                dp[d][i] |= dp[d - 1][i]

                for j in range(n):

                    if f[d][i] and f[d][j]:

                        dp[d][i] |= dp[d - 1][j]

                if len(dp[d][i]) == n:

                    return d

        return -1

    ######################################

    while True:

        n = int(eval(input()))

        if n == 0:

            exit()

        f = [[False] * n for i in range(MAX_DAY + 1)]

        for i in range(n):

            _, *li = list(map(int, input().split()))

            for x in li:

                f[x][i] = True

        # for line in f:

        #     print(line)

        print((solve(n, f)))


problem_p01136()
