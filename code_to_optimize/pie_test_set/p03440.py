def problem_p03440():
    N, M = list(map(int, input().split()))

    A = list(map(int, input().split()))

    XY = [tuple(map(int, input().split())) for i in range(M)]

    if M == N - 1:

        print((0))

        exit()

    class UnionFind:

        def __init__(self, N):

            self.parent = [i for i in range(N)]

            self._size = [1] * N

            self.count = 0

        def root(self, a):

            if self.parent[a] == a:

                return a

            else:

                self.parent[a] = self.root(self.parent[a])

                return self.parent[a]

        def is_same(self, a, b):

            return self.root(a) == self.root(b)

        def unite(self, a, b):

            ra = self.root(a)

            rb = self.root(b)

            if ra == rb:
                return

            if self._size[ra] < self._size[rb]:
                ra, rb = rb, ra

            self._size[ra] += self._size[rb]

            self.parent[rb] = ra

            self.count += 1

        def size(self, a):

            return self._size[self.root(a)]

    uf = UnionFind(N)

    for x, y in XY:

        if uf.is_same(x, y):
            continue

        uf.unite(x, y)

    for i in range(N):

        uf.root(i)

    from collections import defaultdict

    dic = defaultdict(lambda: set())

    for i in range(N):

        dic[uf.root(i)].add(i)

    need = (len(dic) - 1) * 2

    if need > N:

        print("Impossible")

        exit()

    ans = used = 0

    import heapq

    hq = []

    heapq.heapify(hq)

    for r, vs in list(dic.items()):

        mini = -1

        mina = 10**9 + 1

        for i in vs:

            if A[i] < mina:

                mina = A[i]

                mini = i

        ans += mina

        used += 1

        for i in vs:

            if i == mini:
                continue

            heapq.heappush(hq, A[i])

    for _ in range(need - used):

        ans += heapq.heappop(hq)

    print(ans)


problem_p03440()
