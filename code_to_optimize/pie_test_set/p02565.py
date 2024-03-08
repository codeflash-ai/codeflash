def problem_p02565():
    import sys

    class SCC:
        """

        SCC class with non-recursive DFS.

        """

        def __init__(self, N):

            self.N = N

            self.G1 = [[] for _ in range(N)]

            self.G2 = [[] for _ in range(N)]

        def add_edge(self, a, b):

            self.G1[a].append(b)

            self.G2[b].append(a)

        def scc(self):

            self.seen = [0] * self.N

            self.postorder = [-1] * self.N

            self.order = 0

            for i in range(self.N):

                if self.seen[i]:
                    continue

                self._dfs(i)

            self.seen = [0] * self.N

            scclist = []

            for i in self._argsort(self.postorder, reverse=True):

                if self.seen[i]:
                    continue

                cc = self._dfs2(i)

                scclist.append(cc)

            return scclist

        def _argsort(self, arr, reverse=False):

            shift = self.N.bit_length() + 2

            tmp = sorted([arr[i] << shift | i for i in range(len(arr))], reverse=reverse)

            mask = (1 << shift) - 1

            return [tmp[i] & mask for i in range(len(arr))]

        def _dfs(self, v0):

            todo = [~v0, v0]

            while todo:

                v = todo.pop()

                if v >= 0:

                    self.seen[v] = 1

                    for next_v in self.G1[v]:

                        if self.seen[next_v]:
                            continue

                        todo.append(~next_v)

                        todo.append(next_v)

                else:

                    if self.postorder[~v] == -1:

                        self.postorder[~v] = self.order

                        self.order += 1

            return

        def _dfs2(self, v):

            todo = [v]

            self.seen[v] = 1

            cc = [v]

            while todo:

                v = todo.pop()

                for next_v in self.G2[v]:

                    if self.seen[next_v]:
                        continue

                    self.seen[next_v] = 1

                    todo.append(next_v)

                    cc.append(next_v)

            return cc

    class TwoSAT:

        def __init__(self, N):

            self.N = N

            self.scc = SCC(2 * N)

            self.flag = -1

        def add_clause(self, i, f, j, g):

            self.scc.add_edge(f * N + i, (1 ^ g) * N + j)

            self.scc.add_edge(g * N + j, (1 ^ f) * N + i)

        def satisfiable(self):

            if self.flag == -1:

                self.scclist = self.scc.scc()

                self.order = {j: i for i, scc in enumerate(self.scclist) for j in scc}

                self.flag = True

                self.ans = [0] * self.N

                for i in range(self.N):

                    if self.order[i] > self.order[self.N + i]:

                        self.ans[i] = 1

                        continue

                    elif self.order[i] == self.order[i + self.N]:

                        self.flag = False

                        return self.flag

                return self.flag

            else:
                return self.flag

        def answer(self):

            return self.ans

    N, D = map(int, input().split())

    xy = [tuple(map(int, input().split())) for _ in range(N)]

    ts = TwoSAT(N)

    for i in range(N - 1):

        for j in range(i + 1, N):

            for k1, k2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:

                pos1, pos2 = xy[i][k1], xy[j][k2]

                if abs(pos2 - pos1) < D:

                    ts.add_clause(i, k1 ^ 1, j, k2 ^ 1)

    if ts.satisfiable():

        print("Yes")

        print(*[xy[i][ts.ans[i]] for i in range(N)], sep="\n")

    else:

        print("No")


problem_p02565()
