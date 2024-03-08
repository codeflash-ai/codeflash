def problem_p03728():
    N = int(eval(input()))

    P = list(map(int, input().split()))

    if N == 1:

        print((0))

        exit()

    Pi = [0] * (N + 1)

    for i, n in enumerate(P, 1):

        Pi[n] = i

    T = [0] * N

    f = [0] * N

    if Pi[N] > Pi[N - 1]:

        T[N - 1] = 0

        f[N - 1] = N - 1

    else:

        T[N - 1] = 1

        f[N - 1] = N

    for i in range(N - 2, 0, -1):

        if T[i + 1] == 0:

            i_i = Pi[i]

            i_ii = Pi[i + 1]

            if i_ii > i_i:

                T[i] = T[i + 1]

                f[i] = f[i + 1]

            else:

                T[i] = T[i + 1] + 1

                f[i] = i + 1

        else:

            i_f = Pi[f[i + 1]]

            i_i = Pi[i]

            i_ii = Pi[i + 1]

            if i_f < i_i < i_ii or i_ii < i_f < i_i or i_i < i_ii < i_f:

                T[i] = T[i + 1]

                f[i] = f[i + 1]

            else:

                T[i] = T[i + 1] + 1

                f[i] = i + 1

    print((T[1]))


problem_p03728()
