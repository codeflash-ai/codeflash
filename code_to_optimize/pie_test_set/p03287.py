def problem_p03287():
    from collections import defaultdict

    from scipy.misc import comb

    N, M = list(map(int, input().split()))

    A = [int(x) for x in input().split()]

    B = [0] * N

    B[0] = A[0]

    for i in range(1, N):

        B[i] = B[i - 1] + A[i]

    B = [0] + B

    c = defaultdict(int)

    for i in range(N + 1):

        c[B[i] % M] += 1

    ans = 0

    for k, v in list(c.items()):

        if v >= 2:

            ans += comb(v, 2, exact=True)

    print(ans)


problem_p03287()
