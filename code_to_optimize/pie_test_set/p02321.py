def problem_p02321():
    def solve():

        from bisect import bisect_right

        N, W = list(map(int, input().split()))

        a = [tuple(map(int, input().split())) for _ in [0] * N]

        def bitdp(items):

            n = len(items)

            dp = [(0, 0) for _ in [0] * (2**n)]

            for bitset in range(1, 2**n):

                for item_num, (item_v, item_w) in enumerate(items):

                    if bitset & 2**item_num:

                        w, v = dp[bitset - 2**item_num]

                        if w + item_w <= W:

                            dp[bitset] = (w + item_w, v + item_v)

                        break

            dp.sort()

            l, maxv, inf = [(0, 0)], 0, float("inf")

            append = l.append

            for item, (w, v), (nextw, _) in zip(dp, dp, dp[1:] + [(inf, inf)]):

                if w < nextw and maxv < v:

                    append(item)

                    maxv = v

            return l

        dp1 = bitdp(a[: N // 2])

        dp2 = bitdp(a[N // 2 :])

        inf = float("inf")

        result = 0

        for w, v in dp1:

            total = v + dp2[bisect_right(dp2, (W - w, inf)) - 1][1]

            if result < total:

                result = total

        print(result)

    if __name__ == "__main__":

        solve()


problem_p02321()
