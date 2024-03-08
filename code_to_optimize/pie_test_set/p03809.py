def problem_p03809():
    import sys

    input = sys.stdin.readline

    sys.setrecursionlimit(10**7)

    N = int(eval(input()))

    A = [int(x) for x in input().split()]

    UV = [[int(x) - 1 for x in row.split()] for row in sys.stdin.readlines()]

    if N == 2:

        x, y = A

        answer = "YES" if x == y else "NO"

        print(answer)

        exit()

    graph = [[] for _ in range(N)]

    for u, v in UV:

        graph[u].append(v)

        graph[v].append(u)

    def dfs(v, parent=None):

        x = A[v]

        if len(graph[v]) == 1:

            return x

        s = 0

        for w in graph[v]:

            if w == parent:

                continue

            ret = dfs(w, v)

            if ret == None:

                return None

            if x < ret:

                return None

            s += ret

        if 2 * x - s > s:

            return None

        if 2 * x < s:

            return None

        return 2 * x - s

    v = 0

    while len(graph[v]) == 1:

        v += 1

    ret = dfs(v)

    bl = ret == 0

    answer = "YES" if bl else "NO"

    print(answer)


problem_p03809()
