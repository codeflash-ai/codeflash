def problem_p02680():
    import sys

    # from itertools import chain, accumulate

    n, m, *abcdef = list(map(int, sys.stdin.buffer.read().split()))

    ver_lines = []

    hor_lines = []

    x_list = set()

    y_list = set()

    n3 = n * 3

    for a, b, c in zip(abcdef[0:n3:3], abcdef[1:n3:3], abcdef[2:n3:3]):

        y_list.add(a)

        y_list.add(b)

        x_list.add(c)

        ver_lines.append((a, b, c))

    for d, e, f in zip(abcdef[n3 + 0 :: 3], abcdef[n3 + 1 :: 3], abcdef[n3 + 2 :: 3]):

        y_list.add(d)

        x_list.add(e)

        x_list.add(f)

        hor_lines.append((d, e, f))

    x_list.add(0)

    y_list.add(0)

    x_list = sorted(x_list)

    y_list = sorted(y_list)

    x_dict = {x: i for i, x in enumerate(x_list, start=1)}

    y_dict = {y: i for i, y in enumerate(y_list, start=1)}

    row_real = len(x_list)

    col_real = len(y_list)

    row = row_real + 2

    col = col_real + 2

    banned_up_ij = [[0] * row for _ in range(col)]

    banned_down_ij = [[0] * row for _ in range(col)]

    banned_left_ij = [[0] * col for _ in range(row)]

    banned_right_ij = [[0] * col for _ in range(row)]

    for a, b, c in ver_lines:

        if a > b:

            a, b = b, a

        ai = y_dict[a]

        bi = y_dict[b]

        j = x_dict[c]

        banned_left_ij[j][ai] += 1

        banned_left_ij[j][bi] -= 1

        banned_right_ij[j - 1][ai] += 1

        banned_right_ij[j - 1][bi] -= 1

    for d, e, f in hor_lines:

        if e > f:

            e, f = f, e

        i = y_dict[d]

        ej = x_dict[e]

        fj = x_dict[f]

        banned_up_ij[i][ej] += 1

        banned_up_ij[i][fj] -= 1

        banned_down_ij[i - 1][ej] += 1

        banned_down_ij[i - 1][fj] -= 1

    banned_up = [0] * (row * col)

    banned_down = [0] * (row * col)

    banned_left = [0] * (row * col)

    banned_right = [0] * (row * col)

    for i in range(col):

        ru = banned_up_ij[i]

        rd = banned_down_ij[i]

        ri = row * i

        banned_up[ri] = ru[0]

        banned_down[ri] = rd[0]

        for j in range(1, row):

            banned_up[ri + j] = banned_up[ri + j - 1] + ru[j]

            banned_down[ri + j] = banned_down[ri + j - 1] + rd[j]

    for j in range(row):

        rl = banned_left_ij[j]

        rr = banned_right_ij[j]

        banned_left[j] = rl[0]

        banned_right[j] = rr[0]

        for i in range(1, col):

            ri0 = (i - 1) * row

            ri1 = i * row

            banned_left[ri1 + j] = banned_left[ri0 + j] + rl[i]

            banned_right[ri1 + j] = banned_right[ri0 + j] + rr[i]

    # banned_up = list(chain.from_iterable(map(accumulate, banned_up_ij)))

    # banned_down = list(chain.from_iterable(map(accumulate, banned_down_ij)))

    # banned_left = list(chain.from_iterable(zip(*map(accumulate, banned_left_ij))))

    # banned_right = list(chain.from_iterable(zip(*map(accumulate, banned_right_ij))))

    # for i in range(col):

    #     print(walls[i * row:(i + 1) * row])

    s = row * y_dict[0] + x_dict[0]

    enable = [-1] * row + ([-1] + [0] * (row - 2) + [-1]) * (col - 2) + [-1] * row

    # for i in range(col):

    #     print(enable[i * row:(i + 1) * row])

    q = [s]

    moves = [(-row, banned_up), (-1, banned_left), (1, banned_right), (row, banned_down)]

    while q:

        c = q.pop()

        if enable[c] == 1:

            continue

        elif enable[c] == -1:

            print("INF")

            exit()

        enable[c] = 1

        for dc, banned in moves:

            if banned[c]:

                continue

            nc = c + dc

            if enable[nc] == 1:

                continue

            q.append(nc)

    # for i in range(col):

    #     print(enable[i * row:(i + 1) * row])

    ans = 0

    for i in range(col):

        ri = i * row

        for j in range(row):

            if enable[ri + j] != 1:

                continue

            t = y_list[i - 1]

            b = y_list[i]

            l = x_list[j - 1]

            r = x_list[j]

            ans += (b - t) * (r - l)

    print(ans)


problem_p02680()
