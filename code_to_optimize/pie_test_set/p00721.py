def problem_p00721():
    from heapq import heappush, heappop

    INF = 10**10

    direct = ((0, -1), (0, 1), (-1, 0), (1, 0))

    def dist(fr, to, mp):

        que = []

        heappush(que, (0, fr))

        visited = [[False] * len(mp[0]) for _ in range(len(mp))]

        visited[fr[1]][fr[0]] = True

        while que:

            d, point = heappop(que)

            x, y = point

            for dx, dy in direct:

                nx, ny = x + dx, y + dy

                if (nx, ny) == to:

                    return d + 1

                if not visited[ny][nx] and mp[ny][nx] != "x":

                    visited[ny][nx] = True

                    heappush(que, (d + 1, (nx, ny)))

        else:

            return -1

    def my_hash(rest):

        return sum([10**i for i in rest])

    def shortest(fr, rest, edges, dp):

        if rest == set():

            return 0

        if my_hash(rest) in dp[fr]:

            return dp[fr][my_hash(rest)]

        ret = INF

        for d, to in edges[fr]:

            if to in rest:

                score = d + shortest(to, rest - {to}, edges, dp)

                if score < ret:

                    ret = score

        dp[fr][my_hash(rest)] = ret

        return ret

    def main():

        while True:

            w, h = list(map(int, input().split()))

            if w == 0:

                break

            mp = ["x" + eval(input()) + "x" for _ in range(h)]

            mp.insert(0, "x" * (w + 2))

            mp.append("x" * (w + 2))

            stains = []

            for y in range(1, h + 1):

                for x in range(1, w + 1):

                    if mp[y][x] == "*":

                        stains.append((x, y))

                    elif mp[y][x] == "o":

                        start = len(stains)

                        stains.append((x, y))

            stain_num = len(stains)

            edges = [[] for _ in range(stain_num)]

            miss_flag = False

            for i in range(stain_num):

                for j in range(i + 1, stain_num):

                    fr = stains[i]

                    to = stains[j]

                    d = dist(fr, to, mp)

                    if d == -1:

                        miss_flag = True

                    edges[i].append((d, j))

                    edges[j].append((d, i))

            if miss_flag:

                print((-1))

                continue

            dp = [{} for _ in range(stain_num)]

            print((shortest(start, {i for i in range(stain_num) if i != start}, edges, dp)))

    main()


problem_p00721()
