def problem_p03451():
    from heapq import heappop, heappush

    n = int(eval(input()))

    grid = [list(map(int, input().split())), list(map(int, input().split()))]

    def biggest(grid, start, cost=0):

        dist = [[0 for _ in range(n)] for _ in range(2)]

        dx = [1, 0]

        dy = [0, 1]

        st = [(cost, start)]

        dist[start[0]][start[1]] = grid[start[0]][start[1]] + cost

        while st:

            c, [sx, sy] = heappop(st)

            for a, b in zip(dx, dy):

                x = sx + a

                y = sy + b

                if x >= 2 or y >= n:

                    continue

                dxy = dist[sx][sy] + grid[x][y]

                if dist[x][y] >= dxy:

                    continue

                dist[x][y] = dxy

                heappush(st, (dist[x][y], [x, y]))

        return dist[-1][-1]

    dist = biggest(grid, [0, 0], cost=0)

    print(dist)


problem_p03451()
