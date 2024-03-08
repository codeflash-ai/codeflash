def problem_p02317():
    import bisect

    N = int(eval(input()))

    A = [int(eval(input())) for _ in range(N)]

    L = [A[0]]

    for a in A[1:]:

        if L[-1] < a:

            L.append(a)

        else:

            L[bisect.bisect_left(L, a)] = a

    print((len(L)))


problem_p02317()
