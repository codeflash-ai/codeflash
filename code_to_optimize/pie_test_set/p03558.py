def problem_p03558():
    from collections import deque

    K = int(eval(input()))

    G = [[] for i in range(K)]

    for n in range(K):

        G[n].append([(n + 1) % K, 1])

        G[n].append([(10 * n) % K, 0])

    que = deque([[1, 1]])

    dist = [float("inf")] * K

    while que:

        node, cost = que.pop()

        dist[node] = min(cost, dist[node])

        for e, e_cost in G[node]:

            if dist[e] != float("inf"):

                continue

            if e_cost == 0:

                que.append([e, cost + 0])

            if e_cost == 1:

                que.appendleft([e, cost + 1])

    print((dist[0]))


problem_p03558()
