def problem_p02368():
    def solve():

        N, M = list(map(int, input().split()))

        edges = [[] for _ in [0] * N]

        r_edges = [[] for _ in [0] * N]

        for _ in [0] * M:

            a, b = list(map(int, input().split()))

            edges[a].append(b)

            r_edges[b].append(a)

        c = get_strongly_connected_components(edges, r_edges)

        group = [0] * N

        for i in range(len(c)):

            for v in c[i]:

                group[v] = i

        result = []

        append = result.append

        for _ in [0] * int(eval(input())):

            a, b = list(map(int, input().split()))

            append("1" if group[a] == group[b] else "0")

        print(("\n".join(result)))

    def get_strongly_connected_components(edges, r_edges):

        import sys

        sys.setrecursionlimit(10**7)

        v_count = len(edges)

        order = [0] * v_count

        k = 1

        def get_order(v):

            order[v] = 1

            nonlocal k

            for dest in edges[v]:

                if order[dest] == 0:

                    get_order(dest)

            order[v] = k

            k += 1

        def get_components(v):

            order[v] = 0

            return [v] + [
                _v for dest in r_edges[v] if order[dest] > 0 for _v in get_components(dest)
            ]

        [None for v in range(v_count) if order[v] == 0 and get_order(v)]

        return [
            get_components(v)
            for v, _ in sorted(enumerate(order), key=lambda x: x[1], reverse=True)
            if order[v] > 0
        ]

    if __name__ == "__main__":

        solve()


problem_p02368()
