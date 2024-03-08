def problem_p03166():
    import sys

    def recursion_g(eges, length, f):

        if length[f] != -1:

            return length[f]

        memo = [0]

        for t in eges[f]:

            memo.append(recursion_g(eges, length, t) + 1)

        length[f] = max(memo)

        return length[f]

    def p_g():

        sys.setrecursionlimit(10**6)

        N, M = list(map(int, input().split()))

        edges = [[] for _ in range(N)]

        length = [-1] * N

        for _ in range(M):

            x, y = list(map(int, input().split()))

            edges[x - 1].append(y - 1)

        for i in range(N):

            recursion_g(edges, length, i)

        print((max(length)))

    if __name__ == "__main__":

        p_g()


problem_p03166()
