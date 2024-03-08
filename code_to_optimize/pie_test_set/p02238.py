def problem_p02238():
    #!/usr/bin/env python

    # -*- coding: utf-8 -*-

    """

    input:

    6

    1 2 2 3

    2 2 3 4

    3 1 5

    4 1 6

    5 1 6

    6 0



    output:

    1 1 12

    2 2 11

    3 3 8

    4 9 10

    5 4 7

    6 5 6

    """

    import sys

    UNVISITED, VISITED_IN_STACK, POPPED_OUT = 0, 1, 2

    def generate_adj_matrix(v_info):

        for v_detail in v_info:

            v_index, adj_num, *adj_list = list(map(int, v_detail))

            # assert len(adj_list) == adj_num

            init_adj_table[v_index].extend(sorted(adj_list, reverse=True))

        return init_adj_table

    def graph_dfs(v_init):

        global time

        # init the first node of overall graph iterations(times >= 1)

        stack.append(v_init)

        d_time[v_init] += time

        # init end

        while stack:

            current = stack[-1]

            v_table = adj_table[current]

            color[current] = VISITED_IN_STACK

            # if adj is None, current's adj(s) have been all visited

            adj = v_table.pop() if v_table else None

            if adj:

                if color[adj] is UNVISITED:

                    color[adj] = VISITED_IN_STACK

                    time += 1

                    d_time[adj] += time

                    stack.append(adj)

            else:

                stack.pop()

                color[current] = POPPED_OUT

                time += 1

                f_time[current] += time

        return None

    def dfs_start():

        global time

        for i in range(vertices):

            if color[i + 1] == UNVISITED:

                # print('round: ', i + 1)

                graph_dfs(i + 1)

                time += 1

    if __name__ == "__main__":

        _input = sys.stdin.readlines()

        vertices = int(_input[0])

        v_info = [x.split() for x in _input[1:]]

        init_adj_table = tuple([] for _ in range(vertices + 1))

        # assert len(vertex_info) == vertex_num

        # config length = (vertex_num + 1)

        stack = []

        color = [UNVISITED] * (vertices + 1)

        d_time, f_time = ([0] * (vertices + 1) for _ in range(2))

        adj_table = generate_adj_matrix(v_info)

        # timing start from 1

        time = 1

        dfs_start()

        for index, v in enumerate(zip(d_time[1:], f_time[1:]), 1):

            print((index, *v))


problem_p02238()
