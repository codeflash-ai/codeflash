def problem_p02324():
    import sys

    f_i = sys.stdin

    V, E = list(map(int, f_i.readline().split()))

    # adjacency matrix

    no_edge = 14001

    adj = [[no_edge] * V for i in range(V)]

    for i in range(V):

        adj[i][i] = 0

    odd_b = 0  # bit DP to record odd vertex

    ans = 0

    # acceptance of input

    for l_i in f_i:

        s, t, d = list(map(int, l_i.split()))

        t_d = min(d, adj[s][t])

        adj[s][t] = t_d

        adj[t][s] = t_d

        odd_b ^= 1 << s

        odd_b ^= 1 << t

        ans += d

    import math

    if odd_b:

        # Warshall???Floyd Algorithm

        for k in range(V):

            for i in range(V):

                for j in range(V):

                    adj[i][j] = min(adj[i][j], adj[i][k] + adj[k][j])

        # Minimum weighted matching

        mw = [no_edge] * (odd_b + 1)

        mw[0] = 0

        digits = int(math.log2(odd_b)) + 1

        for b in range(odd_b):

            for i in range(0, digits):

                if not (b & (1 << i)):

                    for j in range(i + 1, digits):

                        if not (b & (1 << j)):

                            t_b = b + (1 << i) + (1 << j)

                            if t_b == t_b & odd_b:

                                t_w = mw[t_b]

                                mw[t_b] = min(t_w, mw[b] + adj[i][j])

        # Output

        ans += mw[odd_b]

        print(ans)

    else:

        print(ans)


problem_p02324()
