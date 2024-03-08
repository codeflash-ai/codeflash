def problem_p02716():
    N = int(eval(input()))

    A = list(map(int, input().split()))

    INF = 10**18

    dp0 = [-INF] * (N + 2)

    dp1 = [-INF] * (N + 2)

    dp2 = [-INF] * (N + 2)

    dp0[0] = 0

    dp0[1] = 0

    dp1[0] = 0

    dp1[1] = 0

    for i, a in enumerate(A, start=2):

        dp0[i] = dp0[i - 2] + a

        dp1[i] = max(dp1[i - 2] + a, dp0[i - 1])

        dp2[i] = max(dp2[i - 2] + a, dp1[i - 1], dp0[i - 2])

    if N % 2 == 1:

        print((dp2[-1]))

    else:

        print((max(dp1[-1], dp0[-1])))


problem_p02716()
