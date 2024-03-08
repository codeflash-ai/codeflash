def problem_p03395():
    from scipy.sparse.csgraph import dijkstra

    def make_graph(m, subset):

        matrix = [[0] * (m + 1) for _ in range(m + 1)]

        for k in subset:

            for s in range(k, m + 1):

                matrix[s][s % k] = 1

        return dijkstra(matrix)

    def solve(n, aaa, bbb):

        for a, b in zip(aaa, bbb):

            if a < b:

                return -1

        m = max(aaa)

        subset = set(range(1, m + 1))

        sp = make_graph(m, subset)

        if any(sp[a, b] > 50 for a, b in zip(aaa, bbb)):

            return -1

        for k in range(m, 0, -1):

            subset.remove(k)

            sp = make_graph(m, subset)

            if any(sp[a, b] > 50 for a, b in zip(aaa, bbb)):

                subset.add(k)

        return sum(1 << k for k in subset)

    n = int(eval(input()))

    aaa = list(map(int, input().split()))

    bbb = list(map(int, input().split()))

    print((solve(n, aaa, bbb)))


problem_p03395()
