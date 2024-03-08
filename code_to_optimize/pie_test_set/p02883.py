def problem_p02883():
    N, K = list(map(int, input().split()))

    A = list(map(int, input().split()))

    F = list(map(int, input().split()))

    A.sort()

    F.sort(reverse=True)

    # 積をx以下にするために必要な修行回数

    def f(x):

        r = 0

        for i in range(N):

            a, f = A[i], F[i]

            m = a * f

            if m > x:

                r += (m - x + f - 1) // f

        return r

    under = -1

    r = 10**18

    while r - under > 1:

        m = (under + r) // 2

        if f(m) > K:
            under = m

        else:
            r = m

    print(r)


problem_p02883()
