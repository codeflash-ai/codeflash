def problem_p00146():
    def calctime(w, dist):

        time = dist / (2000 / (70.0 + w))

        return time

    n = int(input())

    s = {}

    d = {}

    v = {}

    for i in range(n):

        S, D, V = list(map(int, input().split()))

        s[i] = S

        d[i] = D

        v[i] = V

    dist = [[None] * n for _ in range(n)]

    for i in range(n):

        for j in range(n):

            if i == j:

                dist[i][j] = float("inf")

            else:

                dist[i][j] = abs(d[i] - d[j])

    w = [float("inf")] * (1 << n)

    time = [[float("inf")] * n for _ in range(1 << n)]

    last = [[0] * n for _ in range(1 << n)]

    for i in range(n):

        time[1 << i][i] = 0

        w[1 << i] = v[i] * 20

    for i in range(1, 1 << n):

        for j in range(n):

            if w[i] == float("inf"):
                continue

            for k in range(n):

                if not i >> k & 1:

                    nexti = i | (1 << k)

                    w[nexti] = w[i] + v[k] * 20

                    if time[nexti][k] > calctime(w[i], dist[j][k]) + time[i][j]:

                        time[nexti][k] = calctime(w[i], dist[j][k]) + time[i][j]

                        last[nexti][k] = j

    ans = []

    now = (1 << n) - 1

    last_index = time[now].index(min(time[now]))

    while now != 0:

        ans.append(s[last_index])

        nx = last[now][last_index]

        now = now ^ (1 << last_index)

        last_index = nx

    ans.reverse()

    print((" ".join(map(str, ans))))


problem_p00146()
