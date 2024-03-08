def problem_p00481():
    h, w, n = list(map(int, input().split()))

    stage = [eval(input()) for i in range(h)]

    starts = [str(i) for i in range(n)]

    goals = [str(i + 1) for i in range(n)]

    starts_y = [0 for i in range(n)]

    starts_x = [0 for i in range(n)]

    goals_y = [0 for i in range(n)]

    goals_x = [0 for i in range(n)]

    starts[0] = "S"

    for y in range(h):

        for x in range(w):

            if stage[y][x] in starts:

                starts_y[starts.index(stage[y][x])] = y

                starts_x[starts.index(stage[y][x])] = x

            if stage[y][x] in goals:

                goals_y[goals.index(stage[y][x])] = y

                goals_x[goals.index(stage[y][x])] = x

    move_y = [1, -1, 0, 0]

    move_x = [0, 0, 1, -1]

    sum = 0

    for start_y, start_x, goal_y, goal_x in zip(starts_y, starts_x, goals_y, goals_x):

        bfs_map = [[-1 for j in range(w)] for i in range(h)]

        data_y = [start_y]

        data_x = [start_x]

        bfs_map[start_y][start_x] = 0

        goal = False

        while len(data_y) != 0 and not goal:

            y = data_y.pop(0)

            x = data_x.pop(0)

            goal = False

            for i in range(4):

                y += move_y[i]

                x += move_x[i]

                if y >= 0 and y < h and x >= 0 and x < w:

                    if bfs_map[y][x] == -1 and stage[y][x] != "X":

                        bfs_map[y][x] = bfs_map[y - move_y[i]][x - move_x[i]] + 1

                        data_y.append(y)

                        data_x.append(x)

                    if bfs_map[goal_y][goal_x] != -1:

                        sum += bfs_map[goal_y][goal_x]

                        goal = True

                        break

                y -= move_y[i]

                x -= move_x[i]

    print(sum)


problem_p00481()
