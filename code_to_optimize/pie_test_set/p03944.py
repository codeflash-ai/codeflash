def problem_p03944():
    W, H, N = list(map(int, input().split()))

    graph = [[1] * W for _ in range(H)]

    for _ in range(N):

        x, y, a = list(map(int, input().split()))

        if a == 1:

            for i in range(H):

                for j in range(x):

                    graph[i][j] = 0

        if a == 2:

            for i in range(H):

                for j in range(x, W):

                    graph[i][j] = 0

        if a == 3:

            for i in range(y):

                for j in range(W):

                    graph[i][j] = 0

        if a == 4:

            for i in range(y, H):

                for j in range(W):

                    graph[i][j] = 0

    ans = sum([sum(line) for line in graph])

    print(ans)


problem_p03944()
