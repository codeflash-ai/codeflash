def problem_p00442():
    import sys

    sys.setrecursionlimit(100000)

    def main():

        V, E = int(eval(input())), int(eval(input()))

        L = []

        app = L.append

        visited = [0 for i in range(V)]

        edges = [[] for i in range(V)]

        def visit(x):

            if not visited[x]:

                visited[x] = 1

                for e in edges[x]:

                    visit(e)

                app(x)

        for i in range(E):

            s, t = list(map(int, input().split()))

            edges[s - 1].append(t - 1)

        for i in range(V):

            if not visited[i]:

                visit(i)

        L.reverse()

        flag = 0

        for i in range(V):

            print((L[i] + 1))

            if not flag and i < V - 1 and (L[i + 1] not in edges[L[i]]):

                flag = 1

        print(flag)

    main()


problem_p00442()
