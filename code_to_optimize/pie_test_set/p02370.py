def problem_p02370():
    def main():

        nvertices, nedges = list(map(int, input().split()))

        indegrees = [0 for i in range(nvertices)]

        adj = [[] for i in range(nvertices)]

        for i in range(nedges):

            s, t = list(map(int, input().split()))

            adj[s].append(t)

            indegrees[t] += 1

        S = []

        for i in range(nvertices):

            if indegrees[i] == 0:

                S.append(i)

        ordering = []

        while S:

            u = S.pop(0)

            ordering.append(u)

            for v in adj[u]:

                indegrees[v] -= 1

                if indegrees[v] == 0:

                    S.append(v)

        for v in ordering:

            print(v)

    main()


problem_p02370()
