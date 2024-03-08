def problem_p03171():
    N = int(eval(input()))

    A = list(map(int, input().split()))

    n = N % 2

    DP = [[0] * (N + 1) for _ in range(N + 1)]

    for w in range(1, N + 1):

        for i in range(N):

            j = i + w

            if j > N:

                continue

            if (w + n) % 2 == 1:

                DP[i][j] = min(DP[i + 1][j] - A[i], DP[i][j - 1] - A[j - 1])

            else:

                DP[i][j] = max(DP[i + 1][j] + A[i], DP[i][j - 1] + A[j - 1])

    print((DP[0][N]))


problem_p03171()
