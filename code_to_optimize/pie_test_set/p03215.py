def problem_p03215():
    N, K = list(map(int, input().split()))

    A = list(map(int, input().split()))

    accum = [0] * N

    for i in range(N):

        if i == 0:

            accum[i] = A[i]

        else:

            accum[i] = accum[i - 1] + A[i]

    B = []

    for l in range(N):

        for r in range(l, N):

            if l == 0:

                B.append(accum[r])

            else:

                B.append(accum[r] - accum[l - 1])

    ans = 0

    for i in range(50, -1, -1):

        cnt = 0

        nxt_B = []

        for b in B:

            if b & (1 << i):

                cnt += 1

                nxt_B.append(b)

        if cnt >= K:

            ans += 2**i

            B = nxt_B

    print(ans)


problem_p03215()
