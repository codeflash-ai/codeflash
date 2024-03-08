def problem_p00731():
    from heapq import heappush, heappop

    R_next = tuple((x, y) for x in range(-3, 0) for y in range(-2, 3) if abs(x) + abs(y) <= 3)

    L_next = tuple((x, y) for x in range(1, 4) for y in range(-2, 3) if abs(x) + abs(y) <= 3)

    c_num = tuple(str(i) for i in range(10))

    L, R = 0, 1

    INF = 10**20

    def conv(c):

        if c in c_num:

            return int(c)

        if c in ("T", "S"):

            return 0

        return -1

    while True:

        w, h = list(map(int, input().split()))

        if w == 0:

            break

        mp = [[-1] * 3 + input().split() + [-1] * 3 for _ in range(h)]

        for i in range(h):

            mp[i] = list(map(conv, mp[i]))

        mp.insert(0, [-1] * (w + 6))

        mp.insert(0, [-1] * (w + 6))

        mp.append([-1] * (w + 6))

        mp.append([-1] * (w + 6))

        def search():

            start = []

            goal = []

            for x in range(3, w + 3):

                if mp[h + 1][x] == 0:

                    start.append((x, h + 1))

                if mp[2][x] == 0:

                    goal.append((x, 2))

            que = []

            dic = {}

            for (
                x,
                y,
            ) in start:

                heappush(que, (0, x, y, L))

                heappush(que, (0, x, y, R))

                dic[(x, y, L)] = 0

                dic[(x, y, R)] = 0

            while que:

                total, x, y, foot = heappop(que)

                if foot == R:

                    direct = R_next

                    next_foot = L

                else:

                    direct = L_next

                    next_foot = R

                for dx, dy in direct:

                    nx, ny = x + dx, y + dy

                    if (nx, ny) in goal:

                        return total

                    cost = mp[ny][nx]

                    if cost == -1:

                        continue

                    if not (nx, ny, next_foot) in dic or dic[(nx, ny, next_foot)] > total + cost:

                        dic[(nx, ny, next_foot)] = total + cost

                        heappush(que, (total + cost, nx, ny, next_foot))

            else:

                return -1

        print((search()))


problem_p00731()
