def problem_p03137():
    N, M = list(map(int, input().split()))

    X = sorted(map(int, input().split()))

    d = sorted(j - i for i, j in zip(X, X[1:]))[::-1]

    print((sum(d[N - 1 :])))


problem_p03137()
