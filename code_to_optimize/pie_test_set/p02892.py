def problem_p02892():
    from collections import deque

    N = int(eval(input()))

    S = [list(eval(input())) for _ in range(N)]

    def bfs(i):

        d = [-1 for _ in range(N)]

        d[i] = 0

        que = deque([(i, 0, -1)])

        finished = set()

        while que:

            ci, dist, p = que.popleft()

            # print(ci, dist, p)

            for j in range(N):

                if S[ci][j] == "1":

                    if j != p and d[j] >= 0:

                        if d[j] == dist or (d[j] != dist + 1 and j not in finished):

                            # print(ci, j)

                            return -1

                    elif d[j] == -1:

                        d[j] = dist + 1

                        que.append((j, dist + 1, ci))

            finished.add(ci)

        return dist + 1

    ans = -1

    for i in range(N):

        b = bfs(i)

        ans = max(ans, b)

    print(ans)


problem_p02892()
