def problem_p02366():
    import sys

    sys.setrecursionlimit(10**6)

    def get_articulation_points(G, N, start=0):

        v_min = [0] * N
        order = [None] * N

        result = []
        count = 0

        def dfs(v, prev):

            nonlocal count

            r_min = order[v] = count

            fcnt = 0
            p_art = 0

            count += 1

            for w in G[v]:

                if w == prev:

                    continue

                if order[w] is None:

                    ret = dfs(w, v)

                    p_art |= order[v] <= ret

                    r_min = min(r_min, ret)

                    fcnt += 1

                else:

                    r_min = min(r_min, order[w])

            p_art |= r_min == order[v] and len(G[v]) > 1

            if (prev == -1 and fcnt > 1) or (prev != -1 and p_art):

                result.append(v)

            return r_min

        dfs(start, -1)

        return result

    n, m = map(int, input().split())

    G = [[] for i in range(n)]

    for i in range(m):

        s, t = map(int, input().split())

        G[s].append(t)

        G[t].append(s)

    (*_,) = map(print, sorted(get_articulation_points(G, n)))


problem_p02366()
