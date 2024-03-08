def problem_p03277():
    import itertools

    N = int(eval(input()))

    A = [int(_) for _ in input().split()]

    class SegmentTree:
        """

        Parameters

        ----------

        array : list

            to construct segment tree from

        f : func

            binary operation of the monoid

        e :

            identity element of the monoid

        size : int

            limit for array size

        """

        def __init__(self, array, f, e, size):

            self.f = f

            self.e = e

            self.size = size

            self.n = n = len(array)

            self.dat = [e] * n + array + [e] * (2 * size - 2 * n)

            self.build()

        def build(self):

            dat, n, f = self.dat, self.n, self.f

            for i in range(n - 1, 0, -1):

                dat[i] = f(dat[i << 1], dat[i << 1 | 1])

        def modify(self, p, v):
            """

            set value at position p (0-indexed)

            """

            f, n, dat = self.f, self.n, self.dat

            p += n

            dat[p] = v

            while p > 1:

                dat[p >> 1] = f(dat[p], dat[p ^ 1])

                p >>= 1

        def query(self, l, r):
            """

            result on interval [l, r) (0-indexed)

            """

            f, e, n, dat = self.f, self.e, self.n, self.dat

            res = e

            l += n

            r += n

            while l < r:

                if l & 1:

                    res = f(res, dat[l])

                    l += 1

                if r & 1:

                    r -= 1

                    res = f(res, dat[r])

                l >>= 1

                r >>= 1

            return res

    e = 0

    size = 2 * N + 2

    f = lambda a, b: a + b

    offset = N

    def check(x):

        cumsum = list(itertools.accumulate([0] + [1 if a <= x else -1 for a in A]))

        # sum([l:r)) = cumsum[r] - cumsum[l] >= 0

        # ⇔ cumsum[l] <= cumsum[r]

        ST = SegmentTree([e] * size, f=f, e=e, size=size)

        count = 0

        for c in cumsum:

            c += offset

            count += ST.query(0, c + 1)

            ST.modify(c + 1, ST.query(c + 1, c + 2) + 1)

        return 2 * count > N * (N + 1) // 2

    A_sorted = [0] + sorted(set(A)) + [10**10]

    lb = 0

    rb = len(A_sorted) - 1

    # check(x) == Trueとなる最小のx

    while rb - lb > 1:

        mid = (rb + lb) // 2

        if check(A_sorted[mid]):

            rb = mid

        else:

            lb = mid

    print((A_sorted[rb]))


problem_p03277()
