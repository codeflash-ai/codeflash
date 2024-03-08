def problem_p03176():
    class SegmentTree:

        def __init__(self, a: list, e: "default" = 0, f: "function" = max):

            size = len(a)

            n = 1 << (size - 1).bit_length()

            seg = [e] * (n * 2 - 1)

            for i, a_ in enumerate(a, n - 1):

                seg[i] = a_

            for i in range(n - 2, -1, -1):

                seg[i] = f(seg[i * 2 + 1], seg[i * 2 + 2])

            self.size = size

            self.n = n

            self.seg = seg

            self.a = a

            self.e = e

            self.f = f

        def update(self, k, x):

            n = self.n

            seg = self.seg

            f = self.f

            k += n - 1

            seg[k] = x

            while k > 0:

                k = (k - 1) // 2

                seg[k] = f(seg[k * 2 + 1], seg[k * 2 + 2])

        def query(self, a, b, k=0, l=0, r=-1):

            if r < 0:

                r = self.n

            if r <= a or b <= l:

                return self.e

            if a <= l and r <= b:

                return self.seg[k]

            m = (l + r) // 2

            vl = self.query(a, b, k * 2 + 1, l, m)

            vr = self.query(a, b, k * 2 + 2, m, r)

            return self.f(vl, vr)

    n = int(eval(input()))

    h = tuple(map(int, input().split()))

    a = tuple(map(int, input().split()))

    dp = [0] * (n + 1)

    # dp[j] := 末尾jの単調増加列の美しさの総和の最大値

    seg = SegmentTree(a=dp, e=0, f=max)

    for h_, a_ in zip(h, a):

        # i本目を取って末尾にするとき

        max_ = seg.query(0, h_) + a_

        seg.update(h_, max_)

    print((seg.query(0, n + 1)))


problem_p03176()
