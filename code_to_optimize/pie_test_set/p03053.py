def problem_p03053():
    from collections import deque

    def bfs(maze, visited, queue, H, W):

        count = 0

        while queue:

            y, x = queue.popleft()

            for i, j in ([1, 0], [-1, 0], [0, 1], [0, -1]):

                new_y, new_x = y + i, x + j

                if -1 < new_y < H and -1 < new_x < W:

                    if maze[new_y][new_x] == "." and visited[new_y][new_x] == -1:

                        visited[new_y][new_x] = visited[y][x] + 1

                        queue.append([new_y, new_x])

        ans = -1

        for i in range(H):

            for j in range(W):

                ans = max(ans, visited[i][j])

        return ans

    H, W = list(map(int, input().split()))

    maze = [eval(input()) for i in range(H)]

    visited = [[-1] * W for i in range(H)]

    queue = deque([])

    for i in range(H):

        for j in range(W):

            if maze[i][j] == "#":

                visited[i][j] = 0

                queue.append([i, j])

    print((bfs(maze, visited, queue, H, W)))


problem_p03053()
