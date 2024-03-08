def problem_p00277():
    import sys

    from heapq import heappush, heappop, heapreplace

    def solve():

        file_input = sys.stdin

        N, R, L = list(map(int, file_input.readline().split()))

        pq = [[0, i, 0] for i in range(1, N + 1)]

        m = dict(list(zip(list(range(1, N + 1)), pq)))

        pre_t = 0

        for line in file_input:

            d, t, x = list(map(int, line.split()))

            team = pq[0]

            team[2] += t - pre_t

            pre_t = t

            if team[1] == d:

                team[0] -= x

                if x < 0:

                    heapreplace(pq, team)

            else:

                scored_team = m[d][:]

                scored_team[0] -= x

                heappush(pq, scored_team)

                m[d][2] = -1

                m[d] = scored_team

            while pq[0][2] == -1:

                heappop(pq)

        pq[0][2] += L - pre_t

        ans_team = max(pq, key=lambda x: (x[2], -x[1]))

        print((ans_team[1]))

    solve()


problem_p00277()
