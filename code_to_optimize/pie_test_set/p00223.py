def problem_p00223():
    # from line_profiler import LineProfiler

    from collections import deque

    from sys import stdin

    def main():

        while True:

            W, H = list(map(int, stdin.readline().split()))

            if not W:
                break

            tx, ty = list(map(int, stdin.readline().split()))

            kx, ky = list(map(int, stdin.readline().split()))

            ma = [
                [False] + [bool(1 - int(a)) for a in stdin.readline().split()] + [False]
                for i in range(H)
            ]

            ma = [[False] * (W + 2)] + ma + [[False] * (W + 2)]

            que = deque([[tx, ty, kx, ky, 0]])

            pas = set()

            ans = "NA"

            while que:

                tx, ty, kx, ky, c = que.popleft()

                if tx == kx and ty == ky:

                    ans = c

                    break

                if c + 1 > 100:
                    continue

                for dx, dy in [[-1, 0], [0, -1], [1, 0], [0, 1]]:

                    tdx, tdy = (dx, dy) if ma[ty + dy][tx + dx] else (0, 0)

                    kdx, kdy = (-dx, -dy) if ma[ky - dy][kx - dx] else (0, 0)

                    ttx, tty, kkx, kky = tx + tdx, ty + tdy, kx + kdx, ky + kdy

                    if (ttx, tty, kkx, kky) in pas:
                        continue

                    que.append([ttx, tty, kkx, kky, c + 1])

                    pas.add((ttx, tty, kkx, kky))

            print(ans)

    if __name__ == "__main__":

        # prf = LineProfiler()

        # prf.add_function(main)

        # prf.runcall(main)

        # prf.print_stats()

        main()


problem_p00223()
