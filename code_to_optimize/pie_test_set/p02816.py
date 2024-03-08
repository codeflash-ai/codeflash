def problem_p02816():
    # -*- coding: utf-8 -*-

    import sys

    from collections import Counter, defaultdict

    buff_readline = sys.stdin.buffer.readline

    readline = sys.stdin.readline

    def read_int():

        return int(buff_readline())

    def read_int_n():

        return list(map(int, buff_readline().split()))

    class RollingHash:
        """

        Original code is https://tjkendev.github.io/procon-library/python/string/rolling_hash.html

        """

        class RH:

            def __init__(self, s, base, mod):

                self.base = base

                self.mod = mod

                self.rev = pow(base, mod - 2, mod)

                l = len(s)

                self.h = h = [0] * (l + 1)

                tmp = 0

                for i in range(l):

                    num = s[i]

                    tmp = (tmp * base + num) % mod

                    h[i + 1] = tmp

                self.pw = pw = [1] * (len(s) + 1)

                v = 1

                for i in range(l):

                    pw[i + 1] = v = v * base % mod

            def calc(self, l, r):

                return (self.h[r] - self.h[l] * self.pw[r - l]) % self.mod

        @staticmethod
        def gen(a, b, num):

            result = set()

            while 1:

                import random

                import math

                random.seed()

                while 1:

                    v = random.randint(a, b) // 2 * 2 + 1

                    if v not in result:

                        break

                for x in range(3, int(math.sqrt(v)) + 1, 2):

                    if v % x == 0:

                        break

                else:

                    result.add(v)

                    if len(result) == num:

                        break

            return result

        def __init__(self, s, rand=False, num=5):

            if rand:

                bases = RollingHash.gen(2, 10**3, num)

            else:

                assert num <= 10

                bases = [641, 103, 661, 293, 547, 311, 29, 457, 613, 599][:num]

            MOD = 10**9 + 7

            self.rhs = [self.RH(s, b, MOD) for b in bases]

        def calc(self, l, r):

            return tuple(rh.calc(l, r) for rh in self.rhs)

    def slv(N, A, B):

        a = []

        for i in range(N):

            a.append(A[i] ^ A[i - 1])

        b = []

        for i in range(N):

            b.append(B[i] ^ B[i - 1])

        b = b[:] + b[:]

        ah = RollingHash(a)

        bh = RollingHash(b)

        h = ah.calc(0, N)

        ans = []

        for k in range(N):

            if bh.calc(N - k, N - k + N) == h:

                ans.append("%d %d" % (k, A[0] ^ B[(N - k) % N]))

        return ans

    def main():

        N = read_int()

        A = read_int_n()

        B = read_int_n()

        print(*slv(N, A, B), sep="\n")

    if __name__ == "__main__":

        main()


problem_p02816()
