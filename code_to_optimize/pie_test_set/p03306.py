def problem_p03306():
    n, m = list(map(int, input().split()))

    edge = [[] for i in range(n)]

    for i in range(m):

        u, v, s = list(map(int, input().split()))

        edge[u - 1].append((v - 1, s))

        edge[v - 1].append((u - 1, s))

    ans = [1 for i in range(n)]

    pm = [True] * n

    used = [False] * n

    used[0] = True

    que = [0]

    while que:

        u = que.pop()

        for v, s in edge[u]:

            if not used[v]:

                pm[v] = not pm[u]

                ans[v] = s - ans[u]

                used[v] = True

                que.append(v)

    flag = True

    bipart = True

    for u in range(n):

        for v, s in edge[u]:

            if ans[u] + ans[v] != s:

                flag = False

            if pm[u] == pm[v]:

                bipart = False

                check = (u, v, s)

    if bipart:

        upper = float("inf")

        lower = 1

        for v in range(1, n):

            if pm[v]:

                lower = max(lower, 2 - ans[v])

            else:

                upper = min(upper, ans[v])

        if flag:

            if upper < lower:

                print((0))

            else:

                print((upper - lower + 1))

        else:

            print((0))

    else:

        u, v, s = check

        a, b = ans[u], ans[v]

        if pm[u]:

            diff = s - (ans[u] + ans[v])

            if diff % 2 == 1:

                print((0))

                exit()

            else:

                for i in range(n):

                    if pm[i]:

                        ans[i] += diff // 2

                        if ans[i] < 1:

                            print((0))

                            exit()

                    else:

                        ans[i] -= diff // 2

                        if ans[i] < 1:

                            print((0))

                            exit()

                flag = True

                for u in range(n):

                    for v, s in edge[u]:

                        if ans[u] + ans[v] != s:

                            flag = False

                print((int(flag)))

        else:

            diff = (ans[u] + ans[v]) - s

            if diff % 2 == 1:

                print((0))

                exit()

            else:

                for i in range(n):

                    if pm[i]:

                        ans[i] += diff // 2

                        if ans[i] < 1:

                            print((0))

                            exit()

                    else:

                        ans[i] -= diff // 2

                        if ans[i] < 1:

                            print((0))

                            exit()

                flag = True

                for u in range(n):

                    for v, s in edge[u]:

                        if ans[u] + ans[v] != s:

                            flag = False

                print((int(flag)))


problem_p03306()
