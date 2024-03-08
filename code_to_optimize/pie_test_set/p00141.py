def problem_p00141():
    D = [[-1, 0], [0, 1], [1, 0], [0, -1]]

    def s(N):

        used = [[False] * N + [True] for _ in range(N)] + [[True] * (N + 1)]

        ret = [[" "] * N for _ in range(N)]

        q = [(0, N - 1, 0)]

        used[N - 1][0] = used[N - 1][1] = True

        while len(q) != 0:

            d, y, x = q.pop(0)

            ret[y][x] = "#"

            sy, sx = D[d]

            ry, rx = D[(d + 1) % 4]

            if not used[y + sy][x + sx]:

                q.append((d, y + sy, x + sx))

                used[y + sy][x + sx] = used[y + ry][x + rx] = True

            elif not used[y + ry][x + rx]:

                q.append(((d + 1) % 4, y + ry, x + rx))

                used[y + ry][x + rx] = True

        return ret

    n = eval(input())

    while True:

        n -= 1

        N = eval(input())

        print("\n".join(["".join(a) for a in s(N)]))

        if n == 0:

            break

        print()


problem_p00141()
