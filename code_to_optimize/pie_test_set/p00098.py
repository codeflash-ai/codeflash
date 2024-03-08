def problem_p00098():
    m = -1e9

    n = int(input())

    N = list(range(n))

    B = [[0] * (n + 1) for _ in range(n + 1)]

    for _ in N:

        A = list(map(int, input().split()))

        for j in N:

            s = 0

            b = B[j]

            for k in N[j:]:

                s += A[k]

                b[k] = max(b[k], 0) + s

                m = max(b[k], m)

    print(m)


problem_p00098()
