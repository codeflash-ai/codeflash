def problem_p02769():
    import sys

    sys.setrecursionlimit(10**9)

    MOD = 10**9 + 7

    n, k = list(map(int, input().split()))

    if k >= n - 1:

        k = n - 1

    inv_table = [0] + [1]

    for i in range(2, k + 1):

        inv_table += [inv_table[MOD % i] * (MOD - int(MOD / i)) % MOD]

    comb_sum = 1

    fact = 1

    comb1 = 1

    comb2 = 1

    for i in range(1, k + 1):

        comb1 = (comb1 * (n - i + 1) * inv_table[i]) % MOD

        comb2 = (comb2 * (n - i) * inv_table[i]) % MOD

        comb = (comb1 * comb2) % MOD

        comb_sum += comb

    print((comb_sum % MOD))


problem_p02769()
