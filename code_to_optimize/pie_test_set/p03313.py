def problem_p03313():
    N = int(eval(input()))

    A = [[(0, 0)] * (1 << N) for _ in range(N + 1)]

    A[0] = [(int(a), 0) for a in input().split()]

    combine = lambda a, b: (
        a if a[1] >= b[0] else b if b[1] >= a[0] else (a[0], b[0]) if a[0] > b[0] else (b[0], a[0])
    )

    for i in range(1, N + 1):

        ii = 1 << i - 1

        for j in range(1 << N):

            if j & ii:

                A[i][j] = combine(A[i - 1][j], A[i - 1][j ^ ii])

            else:

                A[i][j] = A[i - 1][j]

    ANS = [0] * (1 << N)

    for i in range(1, 1 << N):

        ANS[i] = max(ANS[i - 1], sum(A[-1][i]))

    print(("\n".join(map(str, ANS[1:]))))


problem_p03313()
