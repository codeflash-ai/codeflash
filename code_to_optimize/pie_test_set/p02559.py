def problem_p02559():
    from numba import njit, b1, i4, i8, f8

    import numpy as np

    import sys

    @njit((i8[:], i8), cache=True)
    def sum(tree, i):

        s = 0

        while i > 0:

            s += tree[i]

            i -= i & -i

        return s

    @njit((i8[:], i8, i8, i8), cache=True)
    def add(tree, size, i, x):

        while i <= size:

            tree[i] += x

            i += i & -i

    @njit((i8[:], i8, i8), cache=True)
    def range_sum(tree, l, r):

        return sum(tree, r) - sum(tree, l)

    def main():

        n, q = list(map(int, sys.stdin.buffer.readline().split()))

        bit = np.array([0] + sys.stdin.buffer.readline().split(), np.int64)

        for i in range(1, n + 1):

            if i + (i & -i) < n + 1:

                bit[i + (i & -i)] += bit[i]

        for y in sys.stdin.buffer.readlines():

            q, p, x = list(map(int, y.split()))

            if q:

                print((range_sum(bit, p, x)))

            else:

                add(bit, n, p + 1, x)

    if __name__ == "__main__":

        main()


problem_p02559()
