def problem_p03143():
    import sys

    input = sys.stdin.readline

    from collections import deque

    class Unionfindtree:

        def __init__(self, number, L):

            self.par = [i for i in range(number)]

            self.rank = [0] * (number)

            self.count = L

        def find(self, x):  # 親を探す

            if self.par[x] == x:

                return x

            else:

                self.par[x] = self.find(self.par[x])

                return self.par[x]

        def union(self, x, y):  # x,yを繋げる

            px = self.find(x)

            py = self.find(y)

            if px == py:

                return

            if self.rank[px] < self.rank[py]:

                self.par[px] = py

                self.count[py] += self.count[px]

            else:

                self.par[py] = px

                self.count[px] += self.count[py]

            if self.rank[px] == self.rank[py]:

                self.rank[px] += 1

        def connect(self, x, y):  # 親が同じかみる

            return self.find(x) == self.find(y)

    N, M = list(map(int, input().split()))

    X = [int(i) for i in input().split()]

    que = []

    table = [[] for i in range(N)]

    for i in range(M):

        a, b, c = list(map(int, input().split()))

        a, b = a - 1, b - 1

        que.append((c, a, b))

    que.sort()

    T = Unionfindtree(N, X)

    ok = deque()

    for i in range(M):

        c, a, b = que[i]

        T.union(a, b)

        if T.count[T.find(a)] >= c:

            ok.append(i)

    for i in range(M):

        c, a, b = que[i]

        table[a].append((b, c, i))

        table[b].append((a, c, i))

    used = [0] * M

    def dfs(i, lim):

        used[i] = 1

        c, a, b = que[i]

        H = deque()

        H.append(b)

        H.append(a)

        while H:

            x = H.popleft()

            for y, c, i in table[x]:

                if used[i] == 1:

                    continue

                if c > lim:

                    continue

                H.append(y)

                used[i] = 1

        return

    while ok:

        i = ok.pop()

        if used[i] == 0:

            c, a, b = que[i]

            dfs(i, c)

    print((M - sum(used)))


problem_p03143()
