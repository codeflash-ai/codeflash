def problem_p03332():
    # coding: utf-8

    # Your code here!

    import array

    SIZE = 300000
    MOD = 998244353

    # inv = array.array('q',[0]*(SIZE+1))# inv[j] = j^{-1} mod M

    # fac = array.array('q',[0]*(SIZE+1))# inv[j] = j! mod M

    # finv = array.array('q',[0]*(SIZE+1))# inv[j] = (j!)^{-1} mod M

    inv = [0] * (SIZE + 1)  # inv[j] = j^{-1} mod M

    fac = [0] * (SIZE + 1)  # inv[j] = j! mod M

    finv = [0] * (SIZE + 1)  # inv[j] = (j!)^{-1} mod M

    inv[1] = 1

    fac[0] = fac[1] = 1

    finv[0] = finv[1] = 1

    for i in range(2, SIZE + 1):

        inv[i] = MOD - (MOD // i) * inv[MOD % i] % MOD

        fac[i] = fac[i - 1] * i % MOD

        finv[i] = finv[i - 1] * inv[i] % MOD

    def choose(n, r):  # nCk mod MOD の計算

        if r < 0 or r > n:

            return 0

        else:

            return (fac[n] * finv[r] % MOD) * finv[n - r] % MOD

    n, a, b, k = [int(i) for i in input().split()]

    ans = 0

    for i in range(n + 1):

        if (k - i * a) % b == 0:

            ans = (ans + choose(n, i) * choose(n, (k - i * a) // b) % MOD) % MOD

    print(ans)


problem_p03332()
