def problem_p00193():
    import sys

    def generate_next_hexes(x, y):

        hexes = list()

        hexes.append((x, y - 1))

        hexes.append((x, y + 1))

        hexes.append((x - 1, y))

        hexes.append((x + 1, y))

        if y % 2:

            hexes.append((x - 1, y - 1))

            hexes.append((x - 1, y + 1))

        else:

            hexes.append((x + 1, y - 1))

            hexes.append((x + 1, y + 1))

        return hexes

    def update_map(hex_map, hexes):

        num_updated_hexes = 0

        distance = 0

        while hexes:

            next_hexes = []

            for pos in hexes:

                x = pos[0]

                y = pos[1]

                if 1 <= x <= m and 1 <= y <= n and (pos not in hex_map or distance < hex_map[pos]):

                    hex_map[pos] = distance

                    num_updated_hexes += 1

                    next_hexes += generate_next_hexes(x, y)

            distance += 1

            hexes = next_hexes

        return num_updated_hexes

    while True:

        (m, n) = [int(i) for i in sys.stdin.readline().split()]

        if m == n == 0:

            break

        s = int(sys.stdin.readline())

        stores = []

        for i in range(s):

            cord = [int(j) for j in sys.stdin.readline().split()]

            stores.append(tuple(cord))

        hex_map = {}

        update_map(hex_map, stores)

        t = int(sys.stdin.readline())

        candidates = []

        for i in range(t):

            cord = [int(j) for j in sys.stdin.readline().split()]

            candidates.append(tuple(cord))

        # search max num

        max_num_blocks = 0

        for candidate in candidates:

            new_hex_map = hex_map.copy()

            num_blocks = update_map(new_hex_map, [candidate])

            max_num_blocks = max(max_num_blocks, num_blocks)

        print(max_num_blocks)


problem_p00193()
