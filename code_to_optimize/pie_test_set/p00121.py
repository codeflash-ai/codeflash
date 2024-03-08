def problem_p00121():
    def to_xyi(str, target):

        index = 0

        for s in list(str):

            if target == int(s):

                x = index % 4

                if index < 4:

                    y = 0

                else:

                    y = 1

                return (x, y, index)

            index += 1

    def zero_swap(str, mx, my):

        zx, zy, zi = to_xyi(str, 0)

        for index in range(len(list(str))):

            sx, sy, si = to_xyi(str, index)

            if (zx + mx) == sx and (zy + my) == sy:

                nstr = []

                for s in list(str):

                    if s == str[zi]:

                        nstr.append(str[si])

                    elif s == str[si]:

                        nstr.append(str[zi])

                    else:

                        nstr.append(s)

                return "".join(nstr)

    rules = (
        (lambda x, y: x <= 2, lambda str: zero_swap(str, 1, 0)),
        (lambda x, y: y == 0, lambda str: zero_swap(str, 0, 1)),
        (lambda x, y: x >= 1, lambda str: zero_swap(str, -1, 0)),
        (lambda x, y: y == 1, lambda str: zero_swap(str, 0, -1)),
    )

    goal = "01234567"

    move_num = {goal: 0}

    queue = [goal]

    while True:

        if len(queue) == 0:
            break

        search_map = queue.pop(0)

        for can_move, move in rules:

            x, y, i = to_xyi(search_map, 0)

            if not can_move(x, y):
                continue

            next_map = move(search_map)

            if next_map in move_num:
                continue

            move_num[next_map] = move_num[search_map] + 1

            queue.append(next_map)

    import sys

    for s in sys.stdin:

        s = s.replace(" ", "").rstrip()

        print(move_num[s])


problem_p00121()
