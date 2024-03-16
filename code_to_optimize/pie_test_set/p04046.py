def problem_p04046(input_data):
    H, W, A, B = list(map(int, input_data.split()))

    def cmb(n, r, mod):

        if r < 0 or r > n:

            return 0

        r = min(r, n - r)

        return g1[n] * g2[r] * g2[n - r] % mod

    mod = 10**9 + 7  # 出力の制限

    n = 3 * 10**5

    g1 = [1, 1]  # 元テーブル

    g2 = [1, 1]  # 逆元テーブル

    inverse = [0, 1]  # 逆元テーブル計算用テーブル

    for i in range(2, n + 1):

        g1.append((g1[-1] * i) % mod)

        inverse.append((-inverse[mod % i] * (mod // i)) % mod)

        g2.append((g2[-1] * inverse[-1]) % mod)

    L = []

    i = H - A

    j = B + 1

    while i > 0 and j <= W:

        L.append((i, j))

        i -= 1

        j += 1

    ans = 0

    for t in L:

        i = t[0]

        j = t[1]

        ans += (cmb(i + j - 2, i - 1, mod) * cmb(H - i + W - j, H - i, mod)) % mod

        ans %= mod

    return ans
