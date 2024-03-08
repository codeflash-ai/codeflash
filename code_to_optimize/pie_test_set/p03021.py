def problem_p03021():
    import sys

    sys.setrecursionlimit(10**4)

    N = int(eval(input()))

    S = eval(input())

    edge = [[] for i in range(N)]

    for i in range(N - 1):

        a, b = list(map(int, input().split()))

        edge[a - 1].append(b - 1)

        edge[b - 1].append(a - 1)

    depth = [0] * N

    num = [0] * N

    sd = [0] * N

    def dfs(v, pv):

        res = depth[v] * (S[v] == "1")

        n = int(S[v])

        for nv in edge[v]:

            if nv != pv:

                depth[nv] = depth[v] + 1

                dfs(nv, v)

                res += sd[nv]

                n += num[nv]

        sd[v] = res

        num[v] = n

    def dfs2(v, pv):

        hosei = depth[v]

        S = sd[v] - num[v] * hosei

        if S % 2 == 0:

            check = 0

            node = -1

            for nv in edge[v]:

                if nv != pv:

                    if sd[nv] - num[nv] * hosei > S // 2:

                        check = sd[nv] - num[nv] * hosei

                        node = nv

            if not check:

                return S // 2

            else:

                minus = dfs2(node, v)

                if check - 2 * minus <= S - check:

                    return S // 2

                else:

                    return S - check + minus

        else:

            check = 0

            node = -1

            k = S // 2

            for nv in edge[v]:

                if nv != pv:

                    if sd[nv] - num[nv] * hosei > k + 1:

                        check = sd[nv] - num[nv] * hosei

                        node = nv

            if not check:

                return k

            else:

                minus = dfs2(node, v)

                if check - 2 * minus <= S - check:

                    return k

                else:

                    return S - check + minus

    ans = 10**18

    for i in range(N):

        depth[i] = 0

        dfs(i, -1)

        # print(i,depth)

        test = dfs2(i, -1)

        if test * 2 == sd[i]:

            ans = min(ans, test)

    if ans == 10**18:

        print((-1))

    else:

        print(ans)


problem_p03021()
