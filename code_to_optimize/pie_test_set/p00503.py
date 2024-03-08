def problem_p00503():
    import sys

    import itertools

    input_lines = sys.stdin.read().splitlines()

    N, K = [int(x) for x in input_lines[0].split(" ")]

    x_grid = set()

    y_grid = set()

    d_grid = set()

    for line in input_lines[1:]:

        x1, y1, d1, x2, y2, d2 = [int(x) for x in line.split(" ")]

        x_grid.add(x1)

        x_grid.add(x2)

        y_grid.add(y1)

        y_grid.add(y2)

        d_grid.add(d1)

        d_grid.add(d2)

    x_grid = sorted(x_grid)

    y_grid = sorted(y_grid)

    d_grid = sorted(d_grid)

    x_grid_index = {x[1]: x[0] for x in enumerate(x_grid)}

    y_grid_index = {y[1]: y[0] for y in enumerate(y_grid)}

    d_grid_index = {d[1]: d[0] for d in enumerate(d_grid)}

    fish_dist = [
        [[0 for i in range(len(d_grid))] for i in range(len(y_grid))] for i in range(len(x_grid))
    ]

    for line in input_lines[1:]:

        x1, y1, d1, x2, y2, d2 = [int(x) for x in line.split(" ")]

        x1_index = x_grid_index[x1]

        x2_index = x_grid_index[x2]

        y1_index = y_grid_index[y1]

        y2_index = y_grid_index[y2]

        d1_index = d_grid_index[d1]

        d2_index = d_grid_index[d2]

        for x, y, d in itertools.product(
            list(range(x1_index, x2_index)),
            list(range(y1_index, y2_index)),
            list(range(d1_index, d2_index)),
        ):

            fish_dist[x][y][d] += 1

    volume = 0

    for x_index, y_index, d_index in itertools.product(
        list(range(len(x_grid))), list(range(len(y_grid))), list(range(len(d_grid)))
    ):

        if fish_dist[x_index][y_index][d_index] >= K:

            x_begin = x_grid[x_index]

            y_begin = y_grid[y_index]

            d_begin = d_grid[d_index]

            x_end = x_grid[x_index + 1]

            y_end = y_grid[y_index + 1]

            d_end = d_grid[d_index + 1]

            volume += (x_end - x_begin) * (y_end - y_begin) * (d_end - d_begin)

    print(volume)


problem_p00503()
