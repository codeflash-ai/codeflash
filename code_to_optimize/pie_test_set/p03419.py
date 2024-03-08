def problem_p03419():
    N, M = list(map(int, input().split()))

    if N == 2 or M == 2:

        print((0))

    else:

        print((abs((N - 2) * (M - 2))))


problem_p03419()
