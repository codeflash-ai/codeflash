def problem_p03682():
    import heapq

    import sys

    input = sys.stdin.readline

    class UnionFind:

        def __init__(self, n):

            self.par = [i for i in range(n + 1)]

            self.rank = [0] * (n + 1)

        def find(self, x):

            if self.par[x] == x:

                return x

            else:

                self.par[x] = self.find(self.par[x])

                return self.par[x]

        def union(self, x, y):

            x = self.find(x)

            y = self.find(y)

            if self.rank[x] < self.rank[y]:

                self.par[x] = y

            else:

                self.par[y] = x

                if self.rank[x] == self.rank[y]:

                    self.rank[x] += 1

        def same_check(self, x, y):

            return self.find(x) == self.find(y)

    def solve():

        n = int(eval(input()))

        query = []

        xquery = []

        yquery = []

        for i in range(n):

            x, y = (int(i) for i in input().split())

            xquery.append((x, y, i))

            yquery.append((y, x, i))

        xquery = sorted(xquery)

        yquery = sorted(yquery)

        for i in range(n - 1):

            tx, ty, place = xquery[i]

            tx2, ty2, place2 = xquery[i + 1]

            cost = min(abs(tx2 - tx), abs(ty2 - ty))

            heapq.heappush(query, (cost, place, place2))

            ty, tx, place = yquery[i]

            ty2, tx2, place2 = yquery[i + 1]

            cost = min(abs(tx2 - tx), abs(ty2 - ty))

            heapq.heappush(query, (cost, place, place2))

        ct = 0

        uf = UnionFind(n)

        while len(query) > 0:

            cost, place1, place2 = heapq.heappop(query)

            if uf.same_check(place1, place2):

                continue

            else:

                uf.union(int(place1), int(place2))

                ct += cost

        print(ct)

    solve()


problem_p03682()
