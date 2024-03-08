def problem_p01336():
    while 1:

        try:

            N, M = list(map(int, input().split()))

            dp = [[0] * (M + 1) for i in range(3)]

            for _ in range(N):

                name = input()

                C, V, D, L = list(map(int, input().split()))

                VDL = [V, D, L]

                if C > M:

                    continue

                for i in range(3):

                    dp[i][C] = max(dp[i][C], VDL[i])

                    for j in range(M):

                        if dp[i][j]:

                            if j + C <= M:

                                dp[i][j + C] = max(dp[i][j + C], dp[i][j] + VDL[i])

            print(max(max(dp[0]), max(dp[1]), max(dp[2])))

        except:

            break


problem_p01336()
