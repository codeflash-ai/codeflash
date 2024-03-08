def problem_p03310():
    N = int(eval(input()))

    A = list(map(int, input().split()))

    S = [0]

    for i in range(N):

        S.append(S[i] + A[i])

    a_before = 1

    c_before = 3

    ans_tmp = []

    for b in range(2, N - 1):

        if S[a_before] > S[b] - S[a_before]:

            a = a_before

        else:

            for i in range(a_before, b):

                if S[i] > S[b] - S[i]:

                    if abs(S[b] - S[i - 1] - S[i - 1]) < abs(S[b] - S[i] - S[i]):

                        a = i - 1

                        a_before = i - 1

                        break

                    else:

                        a = i

                        a_before = i

                        break

            else:

                a = b - 1

                a_before = b - 1

        if S[c_before] - S[b] > S[N] - S[c_before]:

            c = c_before

        else:

            for i in range(c_before, N):

                if S[i] - S[b] > S[N] - S[i]:

                    if abs((S[N] - S[i - 1]) - (S[i - 1] - S[b])) < abs(
                        (S[N] - S[i]) - (S[i] - S[b])
                    ):

                        c = i - 1

                        c_before = i - 1

                        break

                    else:

                        c = i

                        c_before = i

                        break

            else:

                c = N - 1

                c_before = N - 1

        ans_tmp.append(
            max(S[N] - S[c], S[c] - S[b], S[b] - S[a], S[a] - S[0])
            - min(S[N] - S[c], S[c] - S[b], S[b] - S[a], S[a] - S[0])
        )

    print((min(ans_tmp)))


problem_p03310()
