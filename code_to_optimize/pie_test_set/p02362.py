def problem_p02362():
    V, E, r = list(map(int, input().split()))

    g = [[] for i in range(V)]

    for i in range(E):

        s, t, d = list(map(int, input().split()))

        g[s].append((t, d))

    INF = 10**18

    dist = [INF] * V

    dist[r] = 0

    update = 1

    for _ in range(V):

        update = 0

        for v, e in enumerate(g):

            for t, cost in e:

                if dist[v] != INF and dist[v] + cost < dist[t]:

                    dist[t] = dist[v] + cost

                    update = 1

        if not update:

            break

    else:

        print("NEGATIVE CYCLE")

        exit(0)

    for i in range(V):

        print(dist[i] if dist[i] < INF else "INF")


problem_p02362()
