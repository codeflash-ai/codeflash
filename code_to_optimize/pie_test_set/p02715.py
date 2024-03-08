def problem_p02715():
    n, k = list(map(int, input().split()))

    mod = 10**9 + 7

    def power(a, n, mod):

        bi = str(format(n, "b"))  # 2進数

        res = 1

        for i in range(len(bi)):

            res = (res * res) % mod

            if bi[i] == "1":

                res = (res * a) % mod

        return res

    D = [0] * (k + 1)

    ans = 0

    for i in reversed(list(range(1, k + 1))):

        a = k // i

        d = power(a, n, mod)

        j = 1

        while i * j <= k:

            d -= D[i * j]

            j += 1

        D[i] = d

        ans += (d * i) % mod

    print((ans % mod))


problem_p02715()
