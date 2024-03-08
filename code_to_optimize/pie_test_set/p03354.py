def problem_p03354():
    n, m = list(map(int, input().split()))

    ps = list(map(int, input().split()))

    ls = [list(map(int, input().split())) for _ in range(m)]

    class UnionFind:

        def __init__(self, n):

            self.n = n

            self.parents = list(range(n + 1))

            self.ranks = [0 for _ in range(n + 1)]

        def get_root(self, x):

            if self.parents[x] == x:

                return x

            self.parents[x] = self.get_root(self.parents[x])

            return self.parents[x]

        def merge(self, x, y):

            x = self.get_root(x)

            y = self.get_root(y)

            if x != y:

                if self.ranks[x] < self.ranks[y]:

                    self.parents[x] = y

                else:

                    self.parents[y] = x

                    if self.ranks[x] == self.ranks[y]:

                        self.ranks[x] += 1

    uf = UnionFind(n)

    for a, b in ls:

        uf.merge(a, b)

    res = 0

    for i in range(1, n + 1):

        p0 = uf.get_root(i)

        p1 = uf.get_root(ps[i - 1])

        if p0 == p1:

            res += 1

    print(res)


problem_p03354()
