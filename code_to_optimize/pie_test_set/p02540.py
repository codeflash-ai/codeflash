def problem_p02540():
    # Reference: https://note.nkmk.me/python-union-find/

    class UnionFind:

        # if x is root: self.parents[x] = -(the number of the group nodes)

        # else: self.parents[x] = the parent of x

        def __init__(self, n):

            self.n = n

            self.parents = [-1] * n

        # return the parent of x

        def find(self, x):

            history = []

            while self.parents[x] >= 0:

                history.append(x)

                x = self.parents[x]

            for node in history:

                self.parents[node] = x

            return x

        # merge the group of x and the group of y

        def union(self, x, y):

            x = self.find(x)

            y = self.find(y)

            if x == y:

                return

            if self.parents[x] > self.parents[y]:

                x, y = y, x

            self.parents[x] += self.parents[y]

            self.parents[y] = x

        # return the size of the group of x

        def size(self, x):

            return -self.parents[self.find(x)]

        # return whether x and y in a same group

        def same(self, x, y):

            return self.find(x) == self.find(y)

        # return [all nodes which is in the group of x]

        # O(n)

        def members(self, x):

            root = self.find(x)

            return [i for i in range(self.n) if self.find(i) == root]

        # return [all roots]

        # O(n)

        def roots(self):

            return [i for i, x in enumerate(self.parents) if x < 0]

        # return {root: members of the group}

        # O(n * n)

        def all_group_members(self):

            return {r: self.members(r) for r in self.roots()}

    from bisect import bisect

    from sys import stdin

    input = stdin.buffer.readline

    def main():

        n = int(eval(input()))

        l = [0] * n

        for i in range(n):

            x, y = list(map(int, input().split()))

            l[x - 1] = (y - 1, i)

        # iで管理

        uf = UnionFind(n)

        roots = []

        for y, i in l:

            if len(roots) == 0 or -roots[-1][0] > y:

                roots.append((-y, i))

            else:

                ind = bisect(roots, (-y, i))

                new_y = roots[-1][0]

                for old_y, old_i in roots[ind:]:

                    uf.union(i, old_i)

                roots = roots[:ind]

                roots.append((new_y, i))

        for i in range(n):

            print((uf.size(i)))

    main()


problem_p02540()
