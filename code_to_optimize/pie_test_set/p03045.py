def problem_p03045():
    #!/usr/bin/env python3

    # -*- coding: UTF-8 -*-

    class UnionFind:

        def __init__(self, size):

            self.rank = [0] * size

            self.par = [int(i) for i in range(size)]

            self.grp = size

        def find(self, x):

            if x == self.par[x]:
                return x

            self.par[x] = self.find(self.par[x])

            return self.par[x]

        def same(self, x, y):  # 2つの頂点が同じグループであるかを判定する

            return self.find(x) == self.find(y)

        def unite(self, x, y):  # 辺で接続されている2つの頂点を投げて統合する

            x, y = self.find(x), self.find(y)

            if x == y:

                return

            self.grp -= 1

            if self.rank[x] < self.rank[y]:

                self.par[x] = y

            else:

                self.par[y] = x

                if self.rank[x] == self.rank[y]:

                    self.rank[x] += 1

        def group_num(self):

            return self.grp

    n, m = list(map(int, input().split()))

    uf = UnionFind(n)

    for i in range(m):

        x, y, z = list(map(int, input().split()))

        uf.unite(x - 1, y - 1)

    print((uf.grp))


problem_p03045()
