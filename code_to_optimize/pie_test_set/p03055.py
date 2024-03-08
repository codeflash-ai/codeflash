def problem_p03055():
    N = int(eval(input()))

    src = [tuple([int(x) - 1 for x in input().split()]) for i in range(N - 1)]

    es = [[] for i in range(N)]

    for a, b in src:

        es[a].append(b)

        es[b].append(a)

    def farthest_node_and_dist(s):

        dist = [N] * N

        dist[s] = 0

        stack = [s]

        ret_node = -1

        ret_dist = 0

        while stack:

            v = stack.pop()

            d = dist[v]

            for to in es[v]:

                if d + 1 < dist[to]:

                    dist[to] = d + 1

                    stack.append(to)

                    if ret_dist < d + 1:

                        ret_dist = d + 1

                        ret_node = to

        return (ret_node, ret_dist)

    n, d = farthest_node_and_dist(0)

    n, d = farthest_node_and_dist(n)

    print(("Second" if d % 3 == 1 else "First"))


problem_p03055()
