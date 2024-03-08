def problem_p02807():
    n = int(eval(input()))

    x = list(map(int, input().split()))

    def extgcd(a, b):

        r = [1, 0, a]

        w = [0, 1, b]

        while w[2] != 1:

            q = r[2] // w[2]

            r2 = w

            w2 = [r[0] - q * w[0], r[1] - q * w[1], r[2] - q * w[2]]

            r = r2

            w = w2

        # [x,y]

        return [w[0], w[1]]

    def mod_inv(a, mod):

        x = extgcd(a, mod)[0]

        return (mod + x % mod) % mod

    ans = 0

    MOD = 10**9 + 7

    factorial = 1

    for i in range(2, n):

        factorial = (factorial * i % MOD) % MOD

    p = [0] * (n - 1)

    p[0] = 1 * factorial % MOD

    for i in range(1, n - 1):

        p[i] = p[i - 1] + factorial * mod_inv(i + 1, MOD)

        p[i] %= MOD

    for i in range(n - 1):

        ans += (x[i + 1] - x[i]) % MOD * p[i]

        ans %= MOD

    print((int(ans)))


problem_p02807()
