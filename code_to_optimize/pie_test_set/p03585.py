def problem_p03585():
    class Bit:

        # 参考1: http://hos.ac/slides/20140319_bit.pdf

        # 参考2: https://atcoder.jp/contests/arc046/submissions/6264201

        # 検証: https://atcoder.jp/contests/arc046/submissions/7435621

        # values の 0 番目は使わない

        # len(values) を 2 冪 +1 にすることで二分探索の条件を減らす

        def __init__(self, a):

            if hasattr(a, "__iter__"):

                le = len(a)

                self.n = 1 << le.bit_length()  # le を超える最小の 2 冪

                self.values = values = [0] * (self.n + 1)

                values[1 : le + 1] = a[:]

                for i in range(1, self.n):

                    values[i + (i & -i)] += values[i]

            elif isinstance(a, int):

                self.n = 1 << a.bit_length()

                self.values = [0] * (self.n + 1)

            else:

                raise TypeError

        def add(self, i, val):

            n, values = self.n, self.values

            while i <= n:

                values[i] += val

                i += i & -i

        def sum(self, i):  # (0, i]

            values = self.values

            res = 0

            while i > 0:

                res += values[i]

                i -= i & -i

            return res

        def bisect_left(self, v):  # self.sum(i) が v 以上になる最小の i

            n, values = self.n, self.values

            if v > values[n]:

                return None

            i, step = 0, n >> 1

            while step:

                if values[i + step] < v:

                    i += step

                    v -= values[i]

                step >>= 1

            return i + 1

    def inversion_number(arr):

        n = len(arr)

        arr = sorted(list(range(n)), key=lambda x: arr[x])

        bit = Bit(n)

        res = n * (n - 1) >> 1

        for val in arr:

            res -= bit.sum(val + 1)

            bit.add(val + 1, 1)

        return res

    N = int(eval(input()))

    ABC = [list(map(int, input().split())) for _ in range(N)]

    A, B, C = list(zip(*ABC))

    th = N * (N - 1) // 2 // 2 + 1

    def solve(A, B, C):

        # y = (-Ax+C) / B

        if N < 100:

            ok = -1e10

            ng = 1e10

            n_iteration = 70

        else:

            ok = -1e4

            ng = 1e4

            n_iteration = 46

        A, B, C = list(zip(*sorted(zip(A, B, C), key=lambda x: -x[0] / x[1])))

        for _ in range(n_iteration):

            x = (ok + ng) * 0.5

            Y = [(-a * x + c) / b for a, b, c in zip(A, B, C)]

            inv_num = inversion_number(Y)

            if inv_num >= th:

                ok = x

            else:

                ng = x

        return ok

    print((solve(A, B, C), solve(B, A, C)))


problem_p03585()
