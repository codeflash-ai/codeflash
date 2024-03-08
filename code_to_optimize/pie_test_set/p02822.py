def problem_p02822():
    import sys

    input = sys.stdin.readline

    from collections import deque

    N = int(eval(input()))

    X = [[] for i in range(N)]

    for i in range(N - 1):

        x, y = list(map(int, input().split()))

        X[x - 1].append(y - 1)

        X[y - 1].append(x - 1)

    i0 = min([i for i in range(N) if len(X[i]) == 1])

    P = [-1] * N

    Q = deque([i0])

    R = []

    while Q:

        i = deque.popleft(Q)

        R.append(i)

        for a in X[i]:

            if a != P[i]:

                P[a] = i

                X[a].remove(i)

                deque.append(Q, a)

    mod = 10**9 + 7

    inv2 = (mod + 1) // 2

    A = [1] * N

    for i in R[::-1]:

        s = 0

        for j in X[i]:

            A[i] += A[j]

    ans = 0

    for i in range(N):

        s = pow(inv2, N - A[i], mod)

        t = 1 + (1 - pow(inv2, N - A[i], mod)) * pow(2, N - A[i], mod)

        for j in X[i]:

            s = (s * pow(inv2, A[j], mod)) % mod

            t = (t + (1 - pow(inv2, A[j], mod)) * pow(2, A[j], mod)) % mod

        ans = (ans + 1 - s * t) % mod

    print((ans * inv2 % mod))


problem_p02822()
