def problem_p03253(input_data):
    from math import floor, sqrt

    from collections import defaultdict

    def factors(n):

        d = defaultdict(int)

        for i in range(2, floor(sqrt(n)) + 1):

            while n % i == 0:

                n //= i

                d[i] += 1

            if n == 1:

                break

        if n != 1:

            d[n] += 1

        return d

    def inv(x, mod):

        k = mod - 2

        ret = 1

        while k > 0:

            if k & 1:

                ret = (ret * x) % mod

            x = (x * x) % mod

            k >>= 1

        return ret

    N, M = list(map(int, input_data.split()))

    mod = 10**9 + 7

    dic = factors(M)

    K = len(dic)

    SIZE = N + max(dic.values()) if list(dic.values()) else N

    fact = [None] * (SIZE + 1)

    finv = [None] * (SIZE + 1)

    fact[0] = 1

    for i in range(1, SIZE + 1):

        fact[i] = (fact[i - 1] * i) % mod

    finv[SIZE] = inv(fact[SIZE], mod=mod)

    for i in range(SIZE, 0, -1):

        finv[i - 1] = (finv[i] * i) % mod

    def comb(n, k):

        tmp = (finv[k] * finv[n - k]) % mod

        return (fact[n] * tmp) % mod

    ans = 1

    for p in dic:

        ans = (ans * comb(dic[p] + N - 1, dic[p])) % mod

    return ans
