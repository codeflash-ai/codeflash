def problem_p03248():
    import sys

    input = sys.stdin.readline

    S = "-" + input().rstrip()

    N = len(S) - 1

    def solve(S):

        if S[1] == "0":

            return None

        if S[N] == "1":

            return None

        prev = 1

        graph = []

        for n in range(1, N // 2 + 1):

            if S[n] != S[N - n]:

                return None

            if S[n] == "0":

                continue

            for i in range(prev, n):

                graph.append((i, n))

            prev = n

        for i in range(prev, N):

            graph.append((i, N))

        return graph

    graph = solve(S)

    if graph is None:

        print((-1))

    else:

        for x, y in graph:

            print((x, y))


problem_p03248()
