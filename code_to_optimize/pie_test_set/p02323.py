def problem_p02323():
    def solve():

        V, E = list(map(int, input().split()))

        edges = [[] for _ in [0] * V]

        for _ in [0] * E:

            s, t, d = list(map(int, input().split()))

            edges[s].append((t, d))

        result = float("inf")

        beam_width = 80

        for i in range(V):

            q = [(0, i, {i})]

            for j in range(V - 1):

                _q = []

                append = _q.append

                for cost, v, visited in q[: beam_width + 1]:

                    for dest, d_cost in edges[v]:

                        if dest not in visited:

                            append((cost + d_cost, dest, visited | {dest}))

                q = sorted(_q)

            for cost, v, visited in q[: beam_width + 1]:

                for dest, d_cost in edges[v]:

                    if dest == i:

                        if result > cost + d_cost:

                            result = cost + d_cost

                            break

        print((result if result < float("inf") else -1))

    if __name__ == "__main__":

        solve()


problem_p02323()
