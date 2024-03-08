def problem_p03070():
    import math

    import numpy as np

    N = int(eval(input()))

    a = np.array([int(eval(input())) for _ in range(N)])

    s = np.sum(a)

    # Counts of (R >= s / 2)

    dp0 = np.zeros((N + 1, s + 1))

    dp0[0, 0] = 1

    # Counts of (R == s / 2 and G == s / 2)

    dp1 = np.zeros((N + 1, s // 2 + 1))

    dp1[0, 0] = 1

    mod = 998244353

    for i, x in enumerate(a):

        dp0[i + 1, :] += dp0[i, :] * 2 % mod

        dp0[i + 1, x:] += dp0[i, :-x] % mod

        dp1[i + 1, :] += dp1[i, :] % mod

        dp1[i + 1, x:] += dp1[i, :-x] % mod

    c0 = int(np.sum(dp0[N][math.ceil(s / 2) :]))

    c1 = int(dp1[N][s // 2]) if s % 2 == 0 else 0

    ans = (3**N - 3 * c0 + 3 * c1) % mod

    print(ans)


problem_p03070()
