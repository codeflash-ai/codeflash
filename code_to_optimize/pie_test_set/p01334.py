def problem_p01334():
    while True:

        n = int(eval(input()))

        if n == 0:
            break

        to = []

        for i in range(n):

            line = list(map(int, input().split()))

            for j in range(n):

                x, y = line[2 * j : 2 * j + 2]

                to.append(y * n + x)

        order = []

        used = [False] * (n * n)

        def dfs(x):

            if used[x]:
                return

            used[x] = True

            dfs(to[x])

            order.append(x)

        for i in range(n * n):

            dfs(i)

        order.reverse()

        def dfs2(x, used, group):

            if used[x]:
                return False

            if x in group:
                return True

            group.add(x)

            return dfs2(to[x], used, group)

        used = [False] * (n * n)

        ans = 0

        for i in order:

            group = set()

            if not used[i]:

                if dfs2(i, used, group):
                    ans += 1

            for g in group:
                used[g] = True

        print(ans)


problem_p01334()
