def problem_p02343():
    import sys

    def solve():

        n, q = map(int, sys.stdin.readline().split())

        uf = UnionFind(n)

        ans = []

        for lp in range(q):

            c, x, y = map(int, sys.stdin.readline().split())

            if c == 0:

                uf.unite(x, y)

            else:

                ans.append(1 if uf.is_same(x, y) else 0)

        print(*ans, sep="\n")

    class UnionFind:

        def __init__(self, n):

            self.ds = [i for i in range(n)]

            self.root = [i for i in range(n)]

            self.rank = [0] * n

        def find_root(self, x):

            if x != self.root[x]:

                self.root[x] = self.find_root(self.root[x])

            return self.root[x]

        def is_same(self, x, y):

            return self.find_root(x) == self.find_root(y)

        def unite(self, x, y):

            p = self.find_root(x)

            q = self.find_root(y)

            if p == q:

                return None

            if self.rank[p] < self.rank[q]:

                self.root[p] = q

            elif self.rank[q] < self.rank[p]:

                self.root[q] = p

            else:

                self.root[q] = p

                self.rank[p] += 1

    def debug(x, table):

        for name, val in table.items():

            if x is val:

                print("DEBUG:{} -> {}".format(name, val), file=sys.stderr)

                return None

    if __name__ == "__main__":

        solve()


problem_p02343()
