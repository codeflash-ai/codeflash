def problem_p02604():
    import os

    import sys

    import numpy as np

    def solve(n, precalc_x, precalc_y):

        ans = np.full(n + 1, 10**18, dtype=np.int64)

        def get_cost(v, bit):

            cost = 0

            pcx = precalc_x[v]

            pcy = precalc_y[v ^ bit]

            for i in range(n):

                if v & (1 << i):

                    continue

                cost += min(pcx[i], pcy[i])

            return cost

        for bit in range(1 << n):

            k = (bit & 0x5555) + (bit >> 1 & 0x5555)

            k = (k & 0x3333) + (k >> 2 & 0x3333)

            k = (k & 0x0F0F) + (k >> 4 & 0x0F0F)

            k = (k & 0x00FF) + (k >> 8 & 0x00FF)

            v = bit

            while v:

                ans[k] = min(ans[k], get_cost(v, bit))

                v = (v - 1) & bit

            ans[k] = min(ans[k], get_cost(0, bit))

        return ans

    if sys.argv[-1] == "ONLINE_JUDGE":

        from numba.pycc import CC

        cc = CC("my_module")

        cc.export("solve", "(i8, i8[:,:], i8[:,:])")(solve)

        cc.compile()

        exit()

    if os.name == "posix":

        # noinspection PyUnresolvedReferences

        from my_module import solve

    else:

        from numba import njit

        solve = njit("(i8, i8[:,:], i8[:,:])", cache=True)(solve)

        print("compiled", file=sys.stderr)

    inp = np.fromstring(sys.stdin.read(), dtype=np.int64, sep=" ")

    n = inp[0]

    xxx = inp[1::3]

    yyy = inp[2::3]

    ppp = inp[3::3]

    bit_d = (((np.arange(1 << n)[:, None] & (1 << np.arange(n)))) > 0).astype(np.int64)

    precalc_x = (
        abs((xxx[None, :] * bit_d)[..., None] - xxx[None, None, :]).min(axis=1) * ppp[None, :]
    )

    precalc_y = (
        abs((yyy[None, :] * bit_d)[..., None] - yyy[None, None, :]).min(axis=1) * ppp[None, :]
    )

    ans = solve(n, precalc_x, precalc_y)

    print("\n".join(map(str, ans)))


problem_p02604()
