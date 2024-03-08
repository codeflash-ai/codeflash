def problem_p04034():
    N, M = list(map(int, input().split()))

    R = [0 for _ in range(N + 1)]

    R[1] = 1

    B = [1 for _ in range(N + 1)]

    B[0] = 0

    for _ in range(M):

        x, y = list(map(int, input().split()))

        if R[x] == 1:

            R[y] = 1

        B[x] -= 1

        B[y] += 1

        if B[x] == 0:

            R[x] = 0

    print((sum(R)))


problem_p04034()
