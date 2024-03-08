def problem_p00768():
    while 1:

        M, T, P, R = list(map(int, input().split()))

        if M == 0:
            break

        S = [[-1] * P for i in range(T)]

        PN = [[0] * P for i in range(T)]

        for i in range(R):

            m, t, p, j = list(map(int, input().split()))

            if j == 0:

                S[t - 1][p - 1] = m + PN[t - 1][p - 1]

            else:

                PN[t - 1][p - 1] += 20

        A = [[t + 1, P - S[t].count(-1), sum(S[t])] for t in range(T)]

        A = sorted(
            sorted(sorted(A, reverse=True), key=lambda x: x[2]), key=lambda x: x[1], reverse=True
        )

        ans = str(A[0][0])

        for i in range(1, T):

            if A[i - 1][1] == A[i][1] and A[i - 1][2] == A[i][2]:

                ans += "=" + str(A[i][0])

            else:

                ans += "," + str(A[i][0])

        print(ans)


problem_p00768()
