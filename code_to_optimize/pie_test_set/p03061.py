def problem_p03061():
    import sys

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    class SegmentTree:

        def __init__(self, N, func, I):

            self.func = func

            self.I = I

            self.N = N

            self.sz = 2 ** (N - 1).bit_length()

            self.seg = [I] * (self.sz * 2)

        def assign(self, k, x):

            self.seg[k + self.sz] = x

        def build(self):

            for i in reversed(list(range(1, self.sz))):

                self.seg[i] = self.func(self.seg[2 * i], self.seg[2 * i + 1])

        def update(self, k, x):

            k += self.sz

            self.seg[k] = x

            while k > 1:

                k >>= 1

                self.seg[k] = self.func(self.seg[2 * k], self.seg[2 * k + 1])

        def query(self, a, b):

            L = self.I

            R = self.I

            a += self.sz

            b += self.sz

            while a < b:

                if a & 1:

                    L = self.func(L, self.seg[a])

                    a += 1

                if b & 1:

                    b -= 1

                    R = self.func(self.seg[b], R)

                a >>= 1

                b >>= 1

            return self.func(L, R)

    def main():

        def gcd(a, b):

            while b:

                a, b = b, a % b

            return a

        N = int(readline())

        L = list(map(int, readline().split()))

        seg = SegmentTree(N, gcd, 0)

        for i, n in enumerate(L):

            seg.assign(i, n)

        seg.build()

        ans = 1

        for i in range(N):

            ans = max(ans, gcd(seg.query(0, i), seg.query(i + 1, N)))

        print(ans)

    main()


problem_p03061()
