def problem_p02746():
    def solve(a, b, c, d):

        for k in range(29, -1, -1):

            block_size = 3**k

            box_size = block_size * 3

            x1, y1, x2, y2 = [x // block_size for x in (a, b, c, d)]

            x1, x2 = sorted([x1, x2])

            y1, y2 = sorted([y1, y2])

            if x1 != x2 and y1 != y2:

                return abs(a - c) + abs(b - d)

            if x1 == x2 and x1 % 3 == 1 and (y1 + 1) // 3 * 3 + 1 < y2:

                a, c = [x % box_size for x in (a, c)]

                return min(a + c - 2 * block_size + 2, 4 * block_size - a - c) + abs(b - d)

            elif y1 == y2 and y1 % 3 == 1 and (x1 + 1) // 3 * 3 + 1 < x2:

                b, d = [x % box_size for x in (b, d)]

                return min(b + d - 2 * block_size + 2, 4 * block_size - b - d) + abs(a - c)

        return abs(a - c) + abs(b - d)

    for i in range(int(eval(input()))):

        print((solve(*[int(x) - 1 for x in input().split()])))


problem_p02746()
