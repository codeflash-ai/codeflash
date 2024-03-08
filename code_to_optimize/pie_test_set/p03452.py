def problem_p03452():
    class WeightedUnionFind:

        def __init__(self, n):

            self.parents = [-1] * n

            self.par_weight = [0] * n

        def find(self, x):

            if self.parents[x] < 0:

                return x

            else:

                root = self.find(self.parents[x])

                self.par_weight[x] += self.par_weight[self.parents[x]]

                self.parents[x] = root

                return self.parents[x]

        def union(self, x, y, w):

            w = w + self.weight(x) - self.weight(y)

            x = self.find(x)

            y = self.find(y)

            if x == y:

                return

            if self.parents[x] > self.parents[y]:

                self.parents[y] += self.parents[x]

                self.parents[x] = y

                self.par_weight[x] = -w

            else:

                self.parents[x] += self.parents[y]

                self.parents[y] = x

                self.par_weight[y] = w

        def weight(self, x):

            if self.parents[x] < 0:

                return 0

            else:

                return self.par_weight[x] + self.weight(self.parents[x])

        def diff(self, x, y):

            if self.find(x) != self.find(y):

                raise Exception('"{}" belongs to a different tree from "{}"'.format(x, y))

            return self.weight(y) - self.weight(x)

    N, M = list(map(int, input().split()))

    uf = WeightedUnionFind(N)

    for i in range(M):

        L, R, D = list(map(int, input().split()))

        L -= 1

        R -= 1

        if uf.find(L) == uf.find(R):

            if uf.diff(L, R) != D:

                res = "No"

                break

        else:

            uf.union(L, R, D)

    else:

        res = "Yes"

    print(res)


problem_p03452()
