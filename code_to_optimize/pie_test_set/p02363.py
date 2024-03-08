def problem_p02363():
    INF = 10**18

    V, E = list(map(int, input().split()))

    cost = [[INF] * V for i in range(V)]

    for i in range(E):

        s, t, d = list(map(int, input().split()))

        cost[s][t] = d

    for i in range(V):

        cost[i][i] = 0

    for k in range(V):

        for i in range(V):

            for j in range(V):

                if cost[i][k] != INF and cost[k][j] != INF:

                    cost[i][j] = min(cost[i][j], cost[i][k] + cost[k][j])

    if any(cost[i][i] < 0 for i in range(V)):

        print("NEGATIVE CYCLE")

    else:

        print("\n".join(" ".join([str(e) if e < INF else "INF" for e in e]) for e in cost))


problem_p02363()
