def problem_p03279():
    from bisect import bisect

    from collections import defaultdict

    class Bit:

        def __init__(self, n, MOD):

            self.size = n

            self.tree = [0] * (n + 1)

            self.depth = n.bit_length()

            self.mod = MOD

        def sum(self, i):

            s = 0

            while i > 0:

                s = (s + self.tree[i]) % self.mod

                i -= i & -i

            return s

        def add(self, i, x):

            while i <= self.size:

                self.tree[i] = (self.tree[i] + x) % self.mod

                i += i & -i

        def debug_print(self):

            for i in range(1, self.size + 1):

                j = (i & -i).bit_length()

                print(("  " * j, self.tree[i]))

        def lower_bound(self, x):

            sum_ = 0

            pos = 0

            for i in range(self.depth, -1, -1):

                k = pos + (1 << i)

                if k <= self.size and sum_ + self.tree[k] < x:

                    sum_ += self.tree[k]

                    pos += 1 << i

            return pos + 1, sum_

    n, m = list(map(int, input().split()))

    xxx = list(map(int, input().split()))

    yyy = list(map(int, input().split()))

    ab = defaultdict(set)

    coordinates = set()

    for x in xxx:

        if x < yyy[0] or yyy[-1] < x:

            continue

        i = bisect(yyy, x)

        a = x - yyy[i - 1]

        b = yyy[i] - x

        ab[a].add(b)

        coordinates.add(b)

    # Bitのindexは1から始まるように作っているが、"0"を取れるようにするため、全体を1ずらす

    cor_dict = {b: i for i, b in enumerate(sorted(coordinates), start=2)}

    cdg = cor_dict.get

    MOD = 10**9 + 7

    bit = Bit(len(coordinates) + 1, MOD)

    bit.add(1, 1)

    for a in sorted(ab):

        bbb = sorted(map(cdg, ab[a]), reverse=True)

        for b in bbb:

            bit.add(b, bit.sum(b - 1))

    print((bit.sum(bit.size)))


problem_p03279()
