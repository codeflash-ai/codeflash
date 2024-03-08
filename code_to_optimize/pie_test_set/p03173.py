def problem_p03173():
    import numpy as np

    N = int(eval(input()))

    A = [0] + list(map(int, input().split()))

    A = np.array(A, dtype=np.int64)

    A = np.cumsum(A)

    dp = np.zeros((N, N), dtype=np.int64)

    for j in range(1, N):

        for i in range(N - j):

            # print (dp[i, i:i + j], dp[i + 1:i + j + 1, i + j])

            tmp = min(dp[i, i : i + j] + dp[i + 1 : i + j + 1, i + j])

            dp[i][i + j] = tmp + A[i + j + 1] - A[i]

    print((dp[0][N - 1]))


problem_p03173()
