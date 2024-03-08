def problem_p03108():
    # -*- coding: utf-8 -*-

    from scipy.misc import comb

    import sys

    sys.setrecursionlimit(1000000000)

    class UnionFind:

        def __init__(self, n):

            self.parent = [i for i in range(n + 1)]

            self.depth = [1] * (n + 1)

            self.count = [1] * (n + 1)

        def find(self, x):

            if self.parent[x] == x:

                if self.depth[x] > 2:

                    self.depth[x] = 2

                return x

            else:

                self.parent[x] = self.find(self.parent[x])

                self.depth[x], self.count[x] = 0, 0

                return self.parent[x]

        def isSame(self, x, y):

            return self.find(x) == self.find(y)

        def union(self, x, y):

            x, y = self.find(x), self.find(y)

            if self.depth[x] < self.depth[y]:

                self.parent[x] = self.parent[y]

                self.count[y] += self.count[x]

                self.depth[x], self.count[x] = 0, 0

            else:

                self.parent[y] = self.parent[x]

                self.count[x] += self.count[y]

                self.depth[y], self.count[y] = 0, 0

                if self.depth[x] == self.depth[y]:

                    self.depth[x] += 1

    N, M = list(map(int, input().split()))

    A, B = [], []

    for _ in range(M):

        a, b = list(map(int, input().split()))

        A.append(a)

        B.append(b)

    A.reverse()

    B.reverse()

    groups = UnionFind(N)

    ans = [comb(N, 2, exact=True)]

    for i in range(M):

        tmp = ans[-1]

        if not groups.isSame(A[i], B[i]):

            nA, nB = groups.count[groups.find(A[i])], groups.count[groups.find(B[i])]

            tmp -= nA * nB

            groups.union(A[i], B[i])

        ans.append(tmp)

    for i in range(M):

        print((ans[-(i + 2)]))


problem_p03108()
