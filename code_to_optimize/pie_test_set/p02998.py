def problem_p02998():
    from collections import Counter

    class UnionFind:

        def __init__(self, size):

            self.data = [-1] * size

        def find(self, x):

            if self.data[x] < 0:

                return x

            else:

                self.data[x] = self.find(self.data[x])

                return self.data[x]

        def union(self, x, y):

            x, y = self.find(x), self.find(y)

            if x != y:

                if self.data[y] < self.data[x]:

                    x, y = y, x

                self.data[x] += self.data[y]

                self.data[y] = x

            return x != y

        def same(self, x, y):

            return self.find(x) == self.find(y)

        def size(self, x):

            return -self.data[self.find(x)]

    MAX = 10**5 + 10

    uf = UnionFind(2 * MAX)

    N, *XY = list(map(int, open(0).read().split()))

    for x, y in zip(*[iter(XY)] * 2):

        uf.union(x, y + MAX)

    X = Counter(uf.find(i) for i in range(MAX))

    Y = Counter(uf.find(i) for i in range(MAX, MAX * 2))

    res = sum(X[i] * Y[i] for i in range(MAX * 2))

    print((res - N))


problem_p02998()
