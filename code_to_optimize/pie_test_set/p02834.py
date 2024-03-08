def problem_p02834():
    n, u, v = list(map(int, input().split()))

    anss = 0

    u -= 1

    v -= 1

    d = [[] for _ in range(n)]

    inf = float("inf")

    aoki = [inf] * n

    for i in range(n - 1):

        a, b = list(map(int, input().split()))

        a -= 1

        b -= 1

        d[a].append(b)

        d[b].append(a)

    Q = d[v]

    aoki[v] = 0

    dis = 1

    visited = {v}

    while Q:

        P = []

        for i in Q:

            visited.add(i)

            for j in d[i]:

                if j not in visited:
                    P.append(j)

            aoki[i] = dis

        dis += 1

        Q = P

    Q = d[u]

    chokudai = [-1] * n

    chokudai[u] = 0

    dis = 1

    visited = {u}

    point = -1

    while Q:

        P = []

        for i in Q:

            visited.add(i)

            for j in d[i]:

                if aoki[j] <= dis + 1:

                    if aoki[j] == dis + 1:
                        anss = 1

                    if point < 0:
                        point = i

                    continue

                if j not in visited:
                    P.append(j)

            chokudai[i] = dis

        dis += 1

        Q = P

    Q = d[point]

    dis = chokudai[point] + 1

    visited = {point}

    while Q:

        P = []

        for i in Q:

            visited.add(i)

            if chokudai[i] == -1:
                continue

            chokudai[i] = max(dis, chokudai[i])

            for j in d[i]:

                if j not in visited:
                    P.append(j)

        dis += 1

        Q = P

    ans = 0

    for i in range(n):

        if aoki[i] > chokudai[i]:
            ans = max(ans, chokudai[i])

    print((ans + anss))

    # print(aoki)

    # print(chokudai)


problem_p02834()
