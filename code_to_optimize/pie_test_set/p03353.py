def problem_p03353():
    S = eval(input())

    k = int(eval(input()))

    A = set()

    for l in range(k + 1):

        for i in range(len(S) - l):

            A.add(S[i : i + l + 1])

    A = sorted(list(A))

    print((A[k - 1]))


problem_p03353()
