def problem_p00991(input_data):
    # AOJ 1501: Grid

    # Python3 2018.7.13 bal4u

    MOD = 100000007

    def dp(n, k):

        if tbl[n][k]:
            return tbl[n][k]

        if (k << 1) > n:
            k = n - k

        if k == 0:
            ans = 1

        elif k == 1:
            ans = n

        else:
            ans = dp(n - 1, k) + dp(n - 1, k - 1)

        tbl[n][k] = ans % MOD

        return tbl[n][k]

    tbl = [[0 for j in range(1001)] for i in range(1001)]

    k = 0

    r, c, a1, a2, b1, b2 = list(map(int, input_data.split()))

    dr = abs(a1 - b1)

    if dr > r - dr:
        dr = r - dr

    if (dr << 1) == r:
        k += 1

    dc = abs(a2 - b2)

    if dc > c - dc:
        dc = c - dc

    if (dc << 1) == c:
        k += 1

    return (dp(dr + dc, min(dr, dc)) << k) % MOD
