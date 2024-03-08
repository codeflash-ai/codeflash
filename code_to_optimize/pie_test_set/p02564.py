def problem_p02564():
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

    def main():

        N, M = list(map(int, input().split()))

        sc = SCC(N)

        for i in range(M):

            a, b = list(map(int, input().split()))

            sc.add_edge(a, b)

        scclist = sc.scc()

        print((len(scclist)))

        for cc in scclist:

            print((len(cc), *cc))

    if __name__ == "__main__":

        main()


problem_p02564()
