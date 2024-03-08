def problem_p02890():
    N = int(eval(input()))

    A = list(map(int, input().split()))

    C = [0] * N

    for a in A:

        C[a - 1] += 1

    D = [0] * (N + 1)

    for c in C:

        D[c] += 1

    Dk_accum = [0] * (N + 1)

    for i in range(1, N + 1):

        Dk_accum[i] = Dk_accum[i - 1] + i * D[i]

    D_sum = sum(D)

    D_accum = [0] * (N + 2)

    D_accum[0] = D_sum

    for i in range(1, N + 1):

        D_accum[i] = D_accum[i - 1] - D[i - 1]

    F = [0] * (N + 1)

    for i in range(1, N + 1):

        F[i] = Dk_accum[i] // i + D_accum[i + 1]

    i = N

    for K in range(1, N + 1):

        while i > 0 and K > F[i]:

            i -= 1

        print(i)


problem_p02890()
