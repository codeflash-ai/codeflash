def problem_p01130():
    while True:

        N, M, s, g1, g2 = list(map(int, input().split()))

        if N == M == s == g1 == g2 == 0:

            break

        s, g1, g2 = s - 1, g1 - 1, g2 - 1

        INF = 10**5

        D = [[INF] * N for _ in range(N)]

        for i in range(N):

            D[i][i] = 0

        for _ in range(M):

            b1, b2, c = list(map(int, input().split()))

            b1, b2 = b1 - 1, b2 - 1

            D[b1][b2] = c

        for k in range(N):

            for i in range(N):

                for j in range(N):

                    D[i][j] = min(D[i][j], D[i][k] + D[k][j])

        print((min(D[s][i] + D[i][g1] + D[i][g2] for i in range(N))))


problem_p01130()
