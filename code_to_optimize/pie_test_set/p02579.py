def problem_p02579():
    #!/usr/bin/env python3

    import collections

    import sys

    input = sys.stdin.readline

    h, w = list(map(int, input().split()))

    sy, sx = list(map(int, input().split()))

    gy, gx = list(map(int, input().split()))

    board = (
        [["#"] * (w + 2)]
        + [list("#" + input().replace("\n", "") + "#") for _ in range(h)]
        + [["#"] * (w + 2)]
    )

    scores = [[10**18] * (w + 2) for _ in range(h + 2)]

    scores[sy][sx] = 0

    q = collections.deque()

    q.append((sy, sx))

    count = 0

    while len(q) != 0:

        y, x = q.popleft()

        for dx, dy in [[-1, 0], [1, 0], [0, -1], [0, 1]]:

            if (
                1 <= x + dx <= w
                and 1 <= y + dy <= h
                and board[y + dy][x + dx] != "#"
                and scores[y + dy][x + dx] > scores[y][x]
            ):

                scores[y + dy][x + dx] = scores[y][x]

                q.appendleft((y + dy, x + dx))

        for dy in range(-2, 3):

            for dx in range(-2, 3):

                if dx == 0 and dy == 0:

                    continue

                if (
                    1 <= x + dx <= w
                    and 1 <= y + dy <= h
                    and board[y + dy][x + dx] != "#"
                    and scores[y + dy][x + dx] > scores[y][x] + 1
                ):

                    scores[y + dy][x + dx] = scores[y][x] + 1

                    q.append((y + dy, x + dx))

    if scores[gy][gx] == 10**18:

        print((-1))

    else:

        print((scores[gy][gx]))


problem_p02579()
