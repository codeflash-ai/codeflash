def problem_p03959():
    N = int(eval(input()))

    T = list(map(int, input().split()))

    A = list(map(int, input().split()))

    T = [T[0]] + T + [T[-1]]

    A = [A[0]] + A + [A[-1]]

    ans = 1

    mod = 10**9 + 7

    for i in range(1, N + 1):

        if T[i - 1] < T[i] and A[i + 1] < A[i]:

            if T[i] != A[i]:

                ans *= 0

            continue

        if T[i - 1] < T[i]:

            if A[i] < T[i]:

                ans *= 0

            continue

        if A[i] > A[i + 1]:

            if T[i] < A[i]:

                ans *= 0

            continue

        if i == 1 or i == N:

            if T[N] != A[1]:

                ans *= 0

            else:

                ans *= 1

            continue

        ans = (ans * min(T[i], A[i])) % mod

    print((ans % mod))


problem_p03959()
