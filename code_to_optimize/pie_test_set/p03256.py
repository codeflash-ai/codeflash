def problem_p03256():
    n, m = list(map(int, input().split()))

    s = eval(input())

    g = [[] for _ in range(n)]

    for _ in range(m):

        a, b = list(map(int, input().split()))

        g[a - 1].append(b - 1)

        g[b - 1].append(a - 1)

    count = [[0, 0] for _ in range(n)]

    bad = []

    for i in range(n):

        for v in g[i]:

            if s[v] == "A":

                count[i][0] += 1

            else:

                count[i][1] += 1

        if count[i][0] * count[i][1] == 0:

            bad.append(i)

    visited = [False] * n

    while bad:

        v = bad.pop()

        if visited[v]:

            continue

        visited[v] = True

        for w in g[v]:

            if not visited[w]:

                if s[v] == "A":

                    count[w][0] -= 1

                else:

                    count[w][1] -= 1

                if count[w][0] * count[w][1] == 0:

                    bad.append(w)

    for i in range(n):

        if count[i][0] * count[i][1] > 0:

            print("Yes")

            exit()

    print("No")


problem_p03256()
