def problem_p03855():
    # UnionFindクラス

    class UnionFind(object):

        def __init__(self, n=1):

            self.par = [i for i in range(n)]

            self.rank = [0 for _ in range(n)]

        def find(self, x):

            if self.par[x] == x:

                return x

            else:

                self.par[x] = self.find(self.par[x])

                return self.par[x]

        # xとyの根を結合させる

        def union(self, x, y):

            x = self.find(x)

            y = self.find(y)

            if x != y:

                if self.rank[x] < self.rank[y]:

                    x, y = y, x

                if self.rank[x] == self.rank[y]:

                    self.rank[x] += 1

                self.par[y] = x

        # x,yの根が同じかどうかを返す（根が同じ:True、根が異なる:False）

        def is_same(self, x, y):

            return self.find(x) == self.find(y)

    n, k, l = list(map(int, input().split()))

    uf_road = UnionFind(n)

    uf_rail = UnionFind(n)

    for i in range(k):

        tmp1, tmp2 = list(map(int, input().split()))

        uf_road.union(tmp1 - 1, tmp2 - 1)

    for i in range(l):

        tmp1, tmp2 = list(map(int, input().split()))

        uf_rail.union(tmp1 - 1, tmp2 - 1)

    memo = {}

    for i in range(n):

        if str(uf_rail.find(i)) + " " + str(uf_road.find(i)) not in memo:

            memo[str(uf_rail.find(i)) + " " + str(uf_road.find(i))] = 1

        else:

            memo[str(uf_rail.find(i)) + " " + str(uf_road.find(i))] += 1

    for i in range(n):

        print((memo[str(uf_rail.find(i)) + " " + str(uf_road.find(i))]))


problem_p03855()
