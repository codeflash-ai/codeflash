def problem_p02376():
    from collections import defaultdict

    def dfs(source, sink, flow, connect):

        if source == sink:

            return flow

        used[source] = 1

        for i, (target, cost, rev_i) in enumerate(connect[source]):

            if not used[target] and cost > 0:

                that_flow = dfs(target, sink, min(flow, cost), connect)

                if that_flow > 0:

                    connect[source][i][1] -= that_flow

                    connect[target][rev_i][1] += that_flow

                    return that_flow

        return 0

    v_num, e_num = (int(n) for n in input().split(" "))

    connect = defaultdict(list)

    for _ in range(e_num):

        s, t, cost = (int(n) for n in input().split(" "))

        connect[s].append([t, cost, len(connect[t])])

        connect[t].append([s, 0, len(connect[s]) - 1])

    answer = 0

    base_used = [0 for n in range(v_num)]

    while True:

        used = base_used.copy()

        now_flow = dfs(0, v_num - 1, float("inf"), connect)

        if now_flow == 0:

            break

        answer += now_flow

    print(answer)


problem_p02376()
