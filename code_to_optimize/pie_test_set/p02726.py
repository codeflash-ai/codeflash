def problem_p02726():
    N, X, Y = list(map(int, input().split()))

    A = [[] for i in range(N + 1)]

    for i in range(1, N):

        for j in range(i, N + 1):

            A[int(min(j - i, abs(X - i) + abs(Y - j) + 1))].append(1)

    for i in range(N):

        if i != 0:

            print((sum(A[i])))


problem_p02726()
