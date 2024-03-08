def problem_p00856():
    while 1:

        N, T, L, B = list(map(int, input().split(" ")))

        if N == T == L == B == 0:
            break

        check = 0

        dp = [[0 for _ in range(N + 1)] for _ in range(T + 1)]

        Lose = set([int(input()) for _ in range(L)])

        Back = set([int(input()) for _ in range(B)])

        dp[0][0] = 1

        for i in range(T):

            for j in range(N):

                rank = i - 1 if j in Lose else i

                for d in range(1, 7):

                    next = j + d

                    if next > N:

                        next = N - (next - N)

                    if next in Back:

                        dp[i + 1][0] += dp[rank][j] / 6.0

                    else:

                        dp[i + 1][next] += dp[rank][j] / 6.0

        print("%6f" % sum([dp[i][N] for i in range(1, T + 1)]))


problem_p00856()
