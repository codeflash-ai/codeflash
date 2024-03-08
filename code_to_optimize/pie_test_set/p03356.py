def problem_p03356():
    import queue

    N, M = list(map(int, input().split()))

    p = [int(i) - 1 for i in input().split()]

    x = [0 for i in range(M)]

    y = [0 for i in range(M)]

    edge = [set() for i in range(N)]

    for i in range(M):

        x[i], y[i] = list(map(int, input().split()))

        x[i] -= 1
        y[i] -= 1

        edge[x[i]].add(y[i])

        edge[y[i]].add(x[i])

    k = 0

    L = [0 for i in range(N)]

    q = queue.Queue()

    reached = [0 for i in range(N)]

    for i in range(N):

        if reached[i] == 1:

            continue

        k += 1

        q.put(i)

        reached[i] = 1

        L[i] = k

        while not (q.empty()):

            r = q.get()

            for v in edge[r]:

                if not (reached[v]):

                    q.put(v)

                    reached[v] = 1

                    L[v] = k

    ans = 0

    for i in range(N):

        if L[i] == L[p[i]]:

            ans += 1

    print(ans)


problem_p03356()
