def problem_p02539():
    ROOT = 3

    MOD = 998244353

    roots = [pow(ROOT, (MOD - 1) >> i, MOD) for i in range(24)]  # 1 の 2^i 乗根

    iroots = [pow(x, MOD - 2, MOD) for x in roots]  # 1 の 2^i 乗根の逆元

    def untt(a, n):

        for i in range(n):

            m = 1 << (n - i - 1)

            for s in range(1 << i):

                w_N = 1

                s *= m * 2

                for p in range(m):

                    a[s + p], a[s + p + m] = (a[s + p] + a[s + p + m]) % MOD, (
                        a[s + p] - a[s + p + m]
                    ) * w_N % MOD

                    w_N = w_N * roots[n - i] % MOD

    def iuntt(a, n):

        for i in range(n):

            m = 1 << i

            for s in range(1 << (n - i - 1)):

                w_N = 1

                s *= m * 2

                for p in range(m):

                    a[s + p], a[s + p + m] = (a[s + p] + a[s + p + m] * w_N) % MOD, (
                        a[s + p] - a[s + p + m] * w_N
                    ) % MOD

                    w_N = w_N * iroots[i + 1] % MOD

        inv = pow((MOD + 1) // 2, n, MOD)

        for i in range(1 << n):

            a[i] = a[i] * inv % MOD

    def convolution(a, b):

        la = len(a)

        lb = len(b)

        deg = la + lb - 2

        n = deg.bit_length()

        if min(la, lb) <= 50:

            if la < lb:

                la, lb = lb, la

                a, b = b, a

            res = [0] * (la + lb - 1)

            for i in range(la):

                for j in range(lb):

                    res[i + j] += a[i] * b[j]

                    res[i + j] %= MOD

            return res

        N = 1 << n

        a += [0] * (N - len(a))

        b += [0] * (N - len(b))

        untt(a, n)

        untt(b, n)

        for i in range(N):

            a[i] = a[i] * b[i] % MOD

        iuntt(a, n)

        return a[: deg + 1]

    def convolution_all(polys):

        if not polys:
            return [1]

        # polys.sort(key=lambda x:len(x))

        N = len(polys)

        height = (N - 1).bit_length()

        N0 = 1 << (height)

        data = [None] * (N0 * 2)

        data[N0 : N0 + N] = polys

        data[N0 + N :] = [[1] for _ in range(N0 - N)]

        # print(data)

        # print(polys)

        for k in range(N0 - 1, 0, -1):

            data[k] = convolution(data[2 * k], data[2 * k + 1])

        return data[1]

    SIZE = 10**5 + 10  # 998244353 #ここを変更する

    inv = [0] * SIZE  # inv[j] = j^{-1} mod MOD

    fac = [0] * SIZE  # fac[j] = j! mod MOD

    finv = [0] * SIZE  # finv[j] = (j!)^{-1} mod MOD

    fac[0] = fac[1] = 1

    finv[0] = finv[1] = 1

    for i in range(2, SIZE):

        fac[i] = fac[i - 1] * i % MOD

    finv[-1] = pow(fac[-1], MOD - 2, MOD)

    for i in range(SIZE - 1, 0, -1):

        finv[i - 1] = finv[i] * i % MOD

        inv[i] = finv[i] * fac[i - 1] % MOD

    def choose(n, r):  # nCk mod MOD の計算

        if 0 <= r <= n:

            return (fac[n] * finv[r] % MOD) * finv[n - r] % MOD

        else:

            return 0

    dfac = [0] * SIZE  # fac[j] = j! mod MOD

    dfac[0] = dfac[1] = 1

    for i in range(2, SIZE):

        dfac[i] = dfac[i - 2] * i % MOD

    import sys

    readline = sys.stdin.readline

    read = sys.stdin.read

    n, *h = list(map(int, read().split()))

    from collections import Counter

    def get(k):  # k 個の頂点からiペア作れる場合の数

        res = [1]

        for i in range(k // 2):

            res.append(res[-1] * choose(k - 2 * i, 2) % MOD)

        for i in range(2, k // 2 + 1):

            res[i] = res[i] * finv[i] % MOD

        return res

    d = Counter(h)

    res = []

    for k, v in list(d.items()):

        if v >= 2:
            res.append(get(v))

        if v > n:

            print((0))

            exit()

    bad = convolution_all(res)

    # print(bad)

    # bad = [1,5,7,3]

    ans = 0

    sgn = 1

    for i, ai in enumerate(bad):

        ans += sgn * (dfac[2 * n - 2 * i - 1] if i < n else 1) * ai

        ans %= MOD

        sgn *= -1

        # print(ans)

    print((ans % MOD))


problem_p02539()
