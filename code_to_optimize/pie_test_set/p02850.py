def problem_p02850():
    from collections import deque

    n = int(eval(input()))

    g = {i: dict() for i in range(n)}

    a_list = [0] * (n - 1)

    b_list = [0] * (n - 1)

    for i in range(n - 1):

        a, b = list(map(int, input().split()))

        a_list[i] = a - 1

        b_list[i] = b - 1

        g[a - 1][b - 1] = -1

        g[b - 1][a - 1] = -1

    k = max([len(g[a]) for a in range(n)])

    used_color = [-1] * n

    used_color[0] = k - 1

    # BFS

    queue = deque([0])

    while len(queue) > 0:

        p = queue.popleft()

        c = used_color[p]

        for q in list(g[p].keys()):

            if used_color[q] != -1:

                continue

            c += 1

            c %= k

            # print(p, q, c)

            g[p][q] = c

            g[q][p] = c

            used_color[q] = c

            queue.append(q)

    print(k)

    for i in range(n - 1):

        print((g[a_list[i]][b_list[i]] + 1))


problem_p02850()
