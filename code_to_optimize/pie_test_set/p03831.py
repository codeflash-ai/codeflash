def problem_p03831():
    N, A, B = list(map(int, input().split()))

    X = list(map(int, input().split()))

    tot = 0

    for i in range(1, N):

        if A * (X[i] - X[i - 1]) <= B:

            tot += A * (X[i] - X[i - 1])

        else:

            tot += B

    print(tot)


problem_p03831()
