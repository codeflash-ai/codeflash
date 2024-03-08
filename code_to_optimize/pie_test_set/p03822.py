def problem_p03822():
    import sys

    sys.setrecursionlimit(202020)

    N = int(eval(input()))

    won = [[] for i in range(N)]

    for i in range(1, N):

        a = int(eval(input())) - 1

        won[a].append(i)

    def height(v):

        if not won[v]:
            return 0

        hs = [height(op) for op in won[v]]

        ret = 0

        for i, h in enumerate(sorted(hs)):

            ret = max(ret, h + len(hs) - i)

        return ret

    print((height(0)))


problem_p03822()
