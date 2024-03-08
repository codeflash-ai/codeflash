def problem_p02728():
    import sys

    input = sys.stdin.readline

    mod = pow(10, 9) + 7

    sys.setrecursionlimit(pow(10, 8))

    def mul(a, b):

        return ((a % mod) * (b % mod)) % mod

    def div(a, b):

        return mul(a, modinv(b))

    def modinv(a):

        b, u, v = mod, 1, 0

        while b:

            t = a // b

            a, u = a - t * b, u - t * v

            a, b, u, v = b, a, v, u

        u %= mod

        return u

    def cmb(n, r, mod):

        if r < 0 or r > n:

            return 0

        r = min(r, n - r)

        return g1[n] * g2[r] * g2[n - r] % mod

    NNN = 2 * 10**5

    g1 = [1, 1]

    g2 = [1, 1]

    inverse = [0, 1]

    for i in range(2, NNN + 1):

        g1.append((g1[-1] * i) % mod)

        inverse.append((-inverse[mod % i] * (mod // i)) % mod)

        g2.append((g2[-1] * inverse[-1]) % mod)

    (N,) = list(map(int, input().split()))

    d = [list() for _ in range(N + 1)]

    for _ in range(N - 1):

        a, b = list(map(int, input().split()))

        d[a].append(b)

        d[b].append(a)

    vs = set([1])

    stack = [1]

    vs_bfs = list()

    parents = [0] * (N + 1)

    while stack:

        v = stack.pop()

        vs_bfs.append(v)

        for u in d[v]:

            if u in vs:

                continue

            parents[u] = v

            vs.add(u)

            stack.append(u)

    dp1 = [0 for _ in range(N + 1)]

    sss = [0 for _ in range(N + 1)]

    for v in vs_bfs[::-1]:

        t = 1

        ts = []

        for u in d[v]:

            if u == parents[v]:

                continue

            t = mul(dp1[u], t)

            ts.append(sss[u])

        st = sum(ts)

        sss[v] = st + 1

        for tt in ts:

            t = mul(cmb(st, tt, mod), t)

            st -= tt

        dp1[v] = t

    for v in vs_bfs:

        if v == 1:

            continue

        p = parents[v]

        dp1[v] = mul(div(dp1[p], cmb(N - 1, sss[v], mod)), cmb(N - 1, N - sss[v], mod))

    for x in dp1[1:]:

        print(x)


problem_p02728()
