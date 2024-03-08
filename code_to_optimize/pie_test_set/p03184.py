def problem_p03184():
    # coding: utf-8

    # Your code here!

    SIZE = 300000
    MOD = 10**9 + 7  # ここを変更する

    inv = [0] * (SIZE + 1)  # inv[j] = j^{-1} mod MOD

    fac = [0] * (SIZE + 1)  # fac[j] = j! mod MOD

    finv = [0] * (SIZE + 1)  # finv[j] = (j!)^{-1} mod MOD

    inv[1] = 1

    fac[0] = fac[1] = 1

    finv[0] = finv[1] = 1

    for i in range(2, SIZE + 1):

        inv[i] = MOD - (MOD // i) * inv[MOD % i] % MOD

        fac[i] = fac[i - 1] * i % MOD

        finv[i] = finv[i - 1] * inv[i] % MOD

    def choose(n, r):  # nCk mod MOD の計算

        if 0 <= r <= n:

            return (fac[n] * finv[r] % MOD) * finv[n - r] % MOD

        else:

            return 0

    h, w, n = [int(i) for i in input().split()]

    xy = [[1, 1]] + [[int(i) for i in input().split()] for _ in range(n)]

    dp = [0] * (n + 1)

    dp[0] = 1

    xy.sort(key=lambda x: x[1])

    xy.sort(key=lambda x: x[0])

    # print(xy)

    for i in range(1, n + 1):

        x, y = xy[i]

        for j in range(i):

            xj, yj = xy[j]

            dp[i] -= choose(x - xj + y - yj, y - yj) * dp[j]

            dp[i] %= MOD

    ans = 0

    # print(dp)

    for i, dpi in enumerate(dp):

        x, y = xy[i]

        ans += choose(h - x + w - y, w - y) * dpi

        ans %= MOD

    print(ans)


problem_p03184()
