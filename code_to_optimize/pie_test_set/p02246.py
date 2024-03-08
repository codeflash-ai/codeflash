def problem_p02246():
    from heapq import heappop, heappush

    manhattan = [
        [abs((i % 4) - (j % 4)) + abs((i // 4) - (j // 4)) for j in range(16)] for i in range(16)
    ]

    movables = [
        {1, 4},
        {0, 2, 5},
        {1, 3, 6},
        {2, 7},
        {0, 5, 8},
        {1, 4, 6, 9},
        {2, 5, 7, 10},
        {3, 6, 11},
        {4, 9, 12},
        {5, 8, 10, 13},
        {6, 9, 11, 14},
        {7, 10, 15},
        {8, 13},
        {9, 12, 14},
        {10, 13, 15},
        {11, 14},
    ]

    swap_cache = [[(1 << mf) - (1 << mt) for mt in range(0, 64, 4)] for mf in range(0, 64, 4)]

    destination = 0xFEDCBA9876543210

    def swap(board, move_from, move_to):

        return board + swap_cache[move_from][move_to] * (15 - ((board >> (4 * move_from)) & 15))

    i = 0

    board_init = 0

    blank_init = 0

    for _ in range(4):

        for n in map(int, input().split()):

            if n:

                n -= 1

            else:

                n = 15

                blank_init = i

            board_init += n * 16**i

            i += 1

    estimation_init = sum(
        manhattan[i][((board_init >> (4 * i)) & 15)] for i in range(16) if i != blank_init
    )

    queue = [(estimation_init, board_init, blank_init)]

    visited = set()

    while True:

        estimation, board, blank = heappop(queue)

        if board in visited:

            continue

        elif board == destination:

            print(estimation)

            break

        visited.add(board)

        for new_blank in movables[blank]:

            new_board = swap(board, new_blank, blank)

            if new_board in visited:

                continue

            num = (board >> (4 * new_blank)) & 15

            new_estimation = estimation + 1 - manhattan[new_blank][num] + manhattan[blank][num]

            if new_estimation > 45:

                continue

            heappush(queue, (new_estimation, new_board, new_blank))


problem_p02246()
