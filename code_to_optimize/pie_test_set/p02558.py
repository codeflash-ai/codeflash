def problem_p02558():
    # UnionFind

    import sys

    sys.setrecursionlimit(10**9)

    class UnionFind:

        def __init__(self, n):

            self.n = [-1] * n

            self.r = [0] * n

            self.siz = n

        def find_root(self, x):

            if self.n[x] < 0:

                return x

            else:

                self.n[x] = self.find_root(self.n[x])

                return self.n[x]

        def unite(self, x, y):

            x = self.find_root(x)

            y = self.find_root(y)

            if x == y:

                return

            elif self.r[x] > self.r[y]:

                self.n[x] += self.n[y]

                self.n[y] = x

            else:

                self.n[y] += self.n[x]

                self.n[x] = y

                if self.r[x] == self.r[y]:

                    self.r[y] += 1

            self.siz -= 1

        def root_same(self, x, y):

            return self.find_root(x) == self.find_root(y)

        def count(self, x):

            return -self.n[self.find_root(x)]

        def size(self):

            return self.siz

    n, q = list(map(int, input().split()))

    uf = UnionFind(n)

    for _ in range(q):

        t, u, v = list(map(int, input().split()))

        if t == 0:

            uf.unite(u, v)

        else:

            print((uf.root_same(u, v) + 0))


problem_p02558()
