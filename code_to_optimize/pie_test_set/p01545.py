def problem_p01545():
    import math

    class FenwickTree:

        def __init__(self, a_list, f, default):

            # 0-indexed

            self.N = len(a_list)

            self.bit = a_list[:]

            self.f = f

            self.default = default

            for _ in range(self.N, 1 << (math.ceil(math.log(self.N, 2)))):

                self.bit.append(self.default)

            for i in range(self.N - 1):

                self.bit[i | (i + 1)] = self.f(self.bit[i | (i + 1)], self.bit[i])

        def update(self, i, val):

            while i < self.N:

                self.bit[i] = self.f(self.bit[i], val)

                i |= i + 1

        def query(self, n):

            # [0, n]

            ret = 0

            while n >= 0:

                ret = self.f(ret, self.bit[n])

                n = (n & (n + 1)) - 1

            return ret

    N = int(eval(input()))

    X = [int(x) for x in input().split()]

    dp = FenwickTree([0] * N, lambda x, y: max(x, y), 0)

    for x, i in sorted((x, i) for i, x in enumerate(X)):

        dp.update(i, dp.query(i) + x)

    print((N * (N + 1) // 2 - dp.query(N - 1)))


problem_p01545()
