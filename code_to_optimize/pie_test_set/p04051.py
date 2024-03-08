def problem_p04051():
    import sys

    from collections import defaultdict

    import numpy as np

    def prepare(n, MOD):

        f = 1

        factorials = [1]

        for m in range(1, n + 1):

            f *= m

            f %= MOD

            factorials.append(f)

        inv = pow(f, MOD - 2, MOD)

        invs = [1] * (n + 1)

        invs[n] = inv

        for m in range(n, 1, -1):

            inv *= m

            inv %= MOD

            invs[m - 1] = inv

        return factorials, invs

    MOD = 10**9 + 7

    n = int(sys.stdin.buffer.readline())

    ab = list(map(int, sys.stdin.buffer.read().split()))

    aaa = ab[0::2]

    bbb = ab[1::2]

    max_a = max(aaa)

    max_b = max(bbb)

    facts, invs = prepare(2 * (max_a + max_b), MOD)

    dp_size = max_b * 2 + 1

    ans = 0

    dd_create_func = lambda: np.zeros(dp_size, dtype=np.int64)

    put_ab = defaultdict(dd_create_func)

    get_ab = defaultdict(dd_create_func)

    for a, b in zip(aaa, bbb):

        put_ab[-a][-b + max_b] += 1

        get_ab[a][b + max_b] += 1

        ans = (ans - facts[2 * (a + b)] * invs[2 * a] * invs[2 * b]) % MOD

    dp = np.zeros(dp_size, dtype=np.int64)

    for i in range(-max_a, max_a + 1):

        if i < 0 and i in put_ab:

            dp += put_ab[i]

        dp = np.add.accumulate(dp) % MOD

        if i > 0 and i in get_ab:

            ans = (ans + (dp * get_ab[i]).sum()) % MOD

    print((ans * invs[2] % MOD))


problem_p04051()
