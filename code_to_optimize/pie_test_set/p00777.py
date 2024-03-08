def problem_p00777():
    def solve():

        from collections import deque

        def dfs(start, turn=False):

            path = deque()

            path.append(start)

            bridge_lengths = deque()

            bridge_lengths.append(0)

            unvisited = [True] * (n + 1)

            unvisited[start] = False

            rest = core_islands_num - 1

            diameter = 0

            end_point = start

            while True:

                u = path[-1]

                for i, d in adj_list[u]:

                    if unvisited[i]:

                        path.append(i)

                        unvisited[i] = False

                        rest -= 1

                        bridge_lengths.append(d)

                        break

                else:

                    distance = sum(bridge_lengths)

                    if diameter < distance:

                        diameter = distance

                        end_point = u

                    if rest == 0:

                        break

                    path.pop()

                    bridge_lengths.pop()

            if turn:

                return diameter

            else:

                return end_point

        import sys

        file_input = sys.stdin

        while True:

            n = int(file_input.readline())

            if n == 0:

                break

            p = list(map(int, file_input.readline().split()))

            d = list(map(int, file_input.readline().split()))

            end_bridges_weight = 0

            core_islands_num = n

            adj_list = [[] for i in range(n + 1)]

            s = 1

            for i1, i2, b_l in zip(list(range(2, n + 1)), p, d):

                if i1 not in p[i1 - 1 :]:

                    end_bridges_weight += b_l

                    core_islands_num -= 1

                else:

                    s = i1

                    adj_list[i1].append((i2, b_l))

                    adj_list[i2].append((i1, b_l))

            if p.count(1) == 1:

                del adj_list[2][0]

                end_bridges_weight += d[0]

                core_islands_num -= 1

            e = dfs(s)

            dm = dfs(e, turn=True)

            ans = sum(d) * 3 - end_bridges_weight * 2 - dm

            print(ans)

    solve()


problem_p00777()
