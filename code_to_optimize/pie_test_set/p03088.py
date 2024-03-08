def problem_p03088():
    from itertools import product

    from collections import defaultdict

    MOD = 10**9 + 7

    N = int(eval(input()))

    # XAGC, XGAC, AXGC, AGXC, XACG: prohibited

    A = 0
    C = 1
    G = 2
    T = 3

    cur = defaultdict(lambda: 1)

    cur[(A, G, C)] = cur[(G, A, C)] = cur[(A, C, G)] = 0

    for _ in range(3, N):

        prev = cur

        cur = defaultdict(int)

        for i, j, k, l in product(list(range(4)), repeat=4):

            if (
                (j, k, l) == (A, G, C)
                or (j, k, l) == (G, A, C)
                or (i, k, l) == (A, G, C)
                or (i, j, l) == (A, G, C)
                or (j, k, l) == (A, C, G)
            ):

                continue

            else:

                cur[(j, k, l)] += prev[(i, j, k)]

        for ijk in product(list(range(4)), repeat=3):

            cur[ijk] %= MOD

    ans = sum(cur.values()) % MOD

    if N <= 3:

        print(([1, 4, 16, 61][N]))

    else:

        print(ans)


problem_p03088()
