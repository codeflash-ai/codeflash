def problem_p02682():
    A, B, C, K = list(map(int, input().split()))

    if A >= K:

        print(K)

    elif A + B >= K:

        print(A)

    else:

        print((A - (K - (A + B))))


problem_p02682()
