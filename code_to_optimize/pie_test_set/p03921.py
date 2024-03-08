def problem_p03921():
    # Union-Findデータ構造

    class UnionFind:

        def __init__(self, numV):

            self.pars = list(range(numV))

            self.ranks = [0] * numV

        def find(self, x):

            if self.pars[x] == x:
                return x

            else:

                self.pars[x] = self.find(self.pars[x])

                return self.pars[x]

        def union(self, x, y):

            x, y = self.find(x), self.find(y)

            if x == y:
                return

            if self.ranks[x] < self.ranks[y]:

                self.pars[x] = y

            else:

                self.pars[y] = x

                if self.ranks[x] == self.ranks[y]:

                    self.ranks[x] += 1

        def same(self, x, y):

            return self.find(x) == self.find(y)

    N, M = list(map(int, input().split()))

    UF = UnionFind(N + M)

    for i in range(N):

        K, *Ls = list(map(int, input().split()))

        for L in Ls:

            UF.union(i, L + N - 1)

    pars = [UF.find(i) for i in range(N)]

    print(("YES" if len(set(pars)) == 1 else "NO"))


problem_p03921()
