def problem_p03181():
    import sys

    input = sys.stdin.readline

    sys.setrecursionlimit(10**7)

    N, MOD = list(map(int, input().split()))

    G = [[] for _ in range(N + 1)]

    for _ in range(N - 1):

        x, y = list(map(int, input().split()))

        G[x].append(y)

        G[y].append(x)

    G[0].append(1)

    G[1].append(0)

    DP1 = [-1] * (N + 1)

    DP1[0] = 0

    parent = [-1] * (N + 1)

    parent[1] = 0

    def dfs(x, p):

        tmp = 1

        for next_ in G[x]:

            if next_ == p:

                continue

            parent[next_] = x

            tmp *= dfs(next_, x) + 1

            tmp %= MOD

        DP1[x] = tmp

        return tmp

    dfs(1, 0)

    # print (DP1)

    DP2 = [1] * (N + 1)

    stack = [1]

    # next_とyとで2重ループになってO(N^2) (星型のとき)

    while stack:

        x = stack.pop()

        # 累積積のパート

        # ここが同じ値を返すかチェックする

        n = len(G[x])

        lst1 = [1] * (n + 1)

        lst2 = [1] * (n + 1)

        for i in range(n):

            # 親のときだけは違うルールで対処

            if G[x][i] == parent[x]:

                lst1[i + 1] = lst1[i] * DP2[x] % MOD

            else:

                lst1[i + 1] = lst1[i] * (DP1[G[x][i]] + 1) % MOD

            if G[x][n - 1 - i] == parent[x]:

                lst2[n - i - 1] = lst2[n - i] * DP2[x] % MOD

            else:

                lst2[n - i - 1] = lst2[n - i] * (DP1[G[x][n - 1 - i]] + 1) % MOD

        for i in range(n):

            if G[x][i] == parent[x]:

                continue

            tmp = lst1[i] * lst2[i + 1] % MOD

            tmp += 1  # 白色の分

            DP2[G[x][i]] = tmp % MOD

            stack.append(G[x][i])

    # print (DP2)

    for i in range(1, N + 1):

        print(((DP1[i] * DP2[i]) % MOD))

    # print (DP1)

    # print (DP2)


problem_p03181()
