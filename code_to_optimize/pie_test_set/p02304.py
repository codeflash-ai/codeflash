def problem_p02304():
    # Acceptance of input

    import sys

    file_input = sys.stdin

    n = file_input.readline()

    EP = []

    l = -1000000001

    u = 1000000001

    vs_x = set()

    for line in file_input:

        x1, y1, x2, y2 = list(map(int, line.split()))

        if x1 == x2:

            if y1 < y2:

                EP.append((y1, l, x1))

                EP.append((y2, u, x1))

            else:

                EP.append((y1, u, x1))

                EP.append((y2, l, x1))

            vs_x.add(x1)

        else:

            if x1 < x2:

                EP.append((y1, x1, x2))

            else:

                EP.append((y1, x2, x1))

    vs_x = sorted(vs_x)

    # Binary Indexed Tree

    class BinaryIndexedTree:

        def __init__(self, n):

            self.data = [0] * (n + 1)

            self.num = n

        def switch(self, i, d):

            while i <= self.num:

                self.data[i] += d

                i += i & -i

        def _sum(self, i):

            s = 0

            while i:

                s += self.data[i]

                i -= i & -i

            return s

        def seg_sum(self, a, b):

            return self._sum(b) - self._sum(a - 1)

    # Sweep

    import bisect

    EP.sort()

    BIT = BinaryIndexedTree(len(vs_x))

    cnt = 0

    for p in EP:

        e = p[1]

        if e == l:

            vx = bisect.bisect(vs_x, p[2])

            BIT.switch(vx, 1)

        elif e == u:

            vx = bisect.bisect(vs_x, p[2])

            BIT.switch(vx, -1)

        else:

            l_x = bisect.bisect_left(vs_x, e) + 1

            r_x = bisect.bisect(vs_x, p[2])

            cnt += BIT.seg_sum(l_x, r_x)

    # Output

    print(cnt)


problem_p02304()
