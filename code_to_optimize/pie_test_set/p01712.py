def problem_p01712():
    N, W, H = [int(x) for x in input().split()]

    map_x = [0] * W

    map_y = [0] * H

    for _ in range(N):

        x, y, w = [int(x) for x in input().split()]

        map_x[max(0, x - w)] = max(map_x[max(0, x - w)], x + w - max(0, x - w))

        map_y[max(0, y - w)] = max(map_y[max(0, y - w)], y + w - max(0, y - w))

    def check_wifi(wifi):

        m = 0

        end = len(wifi)

        for i, x in enumerate(wifi):

            if x and i + x > m:

                m = i + x

            if m >= end:

                return True

            if i < m:

                continue

            return False

        return False

    print(("Yes" if check_wifi(map_x) or check_wifi(map_y) else "No"))


problem_p01712()
