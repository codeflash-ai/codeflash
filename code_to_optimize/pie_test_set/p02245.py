def problem_p02245():
    from collections import deque

    from copy import deepcopy, copy

    dy = [-1, 0, 0, 1]

    dx = [0, -1, 1, 0]

    N = 3

    def g(i, j, a):

        t = a // (10**j) % 10

        return a - t * (10**j) + t * (10**i)

    def MAIN():

        m = {
            8: {7, 5},
            7: {8, 6, 4},
            6: {7, 3},
            5: {8, 4, 2},
            4: {7, 5, 3, 1},
            3: {6, 4, 0},
            2: {5, 1},
            1: {4, 2, 0},
            0: {3, 1},
        }

        MAP = "".join(input().replace(" ", "") for _ in range(N))

        start = 8 - MAP.find("0")

        MAP = int(MAP)

        goal = ("1", "2", "3", "4", "5", "6", "7", "8", "0")

        goal = 123456780

        dp = deque([(0, start, MAP)])

        LOG = {MAP}

        while dp:

            cnt, yx, M = dp.popleft()

            if M == goal:

                print(cnt)

                break

            cnt += 1

            for nyx in m[yx]:

                CM = g(yx, nyx, M)

                if not CM in LOG:

                    dp.append((cnt, nyx, CM))

                    LOG.add(CM)

    MAIN()


problem_p02245()
