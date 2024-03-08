def problem_p02919():
    N = int(eval(input()))

    P = list(map(int, input().split()))

    R = list(range(N))

    L = list(range(N))

    I = [-1] * (N + 1)

    for i, p in enumerate(P):

        I[p] = i

    ans = 0

    for n, idx in enumerate(I[1:], 1):

        l = idx - 1

        while l >= 0 and l != L[l]:

            l = L[l]

        r = idx + 1

        while r < N and r != R[r]:

            r = R[r]

        L[idx] = l

        R[idx] = r

        if l != -1:

            l2 = l - 1

            while l2 >= 0 and l2 != L[l2]:

                l2 = L[l2]

            ans += n * (l - l2) * (r - idx)

        if r != N:

            r2 = r + 1

            while r2 < N and r2 != R[r2]:

                r2 = R[r2]

            ans += n * (idx - l) * (r2 - r)

    print(ans)


problem_p02919()
