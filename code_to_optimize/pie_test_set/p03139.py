def problem_p03139():
    N, A, B = list(map(int, input().split()))

    if A + B - N > 0:

        print((min(A, B), A + B - N))

    else:

        print((min(A, B), "0"))


problem_p03139()
