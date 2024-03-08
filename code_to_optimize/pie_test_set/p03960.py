def problem_p03960():
    N, M = list(map(int, input().split()))
    C = [[0] * -~N for _ in range(N + 1)]
    D = list(map(list, C))
    S = [eval(input()) for _ in range(N)]
    r = 0

    for t in range(M - 1):

        for i in range(1, N + 1):

            for j in range(1, N + 1):
                C[i][j] = C[i - 1][j - 1] + int(S[i - 1][t] == S[j - 1][t + 1])
                D[i][j] = min(D[i - 1][j], D[i][j - 1]) + C[i][j]

        r += D[N][N]

    print(r)


problem_p03960()
