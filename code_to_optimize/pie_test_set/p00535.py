def problem_p00535():
    from heapq import heappush, heappop

    h, w = list(map(int, input().split()))

    mp = [[-1] + list(eval(input())) + [-1] for _ in range(h)]

    mp.insert(0, [-1] * (w + 2))

    mp.append([-1] * (w + 2))

    que = []

    for y in range(1, h + 1):

        for x in range(1, w + 1):

            if "1" <= mp[y][x] <= "9":

                mp[y][x] = int(mp[y][x])

            elif mp[y][x] == ".":

                mp[y][x] = 0

                heappush(que, (0, x, y))

    vec = ((0, 1), (0, -1), (1, 1), (1, 0), (1, -1), (-1, 1), (-1, 0), (-1, -1))

    while que:

        turn, x, y = heappop(que)

        for dx, dy in vec:

            nx, ny = x + dx, y + dy

            if mp[ny][nx] > 0:

                mp[ny][nx] -= 1

                if mp[ny][nx] == 0:

                    heappush(que, (turn + 1, nx, ny))

    print(turn)


problem_p00535()
