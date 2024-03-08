def problem_p03722():

    N, M = list(map(int, input().split()))

    edges = []

    for i in range(M):

        a, b, c = list(map(int, input().split()))

        edges.append([a, b, c])

    dist = [float("inf")] * (N + 1)

    dist[1] = 0

    for _ in range(N - 1):

        for a, b, c in edges:

            dist[b] = min(dist[b], dist[a] - c)

    ans = dist[N]

    for a, b, c in edges:

        dist[b] = min(dist[b], dist[a] - c)

    if ans == dist[N]:

        print((-ans))

    else:

        print("inf")


problem_p03722()
