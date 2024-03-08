def problem_p00118():
    import sys

    def solve():

        person = 0

        for i in range(0, H):

            for j in range(0, W):

                if geo[i][j] is not "_":

                    search(i, j)

                    person += 1

        print(person)

    def search(i, j):

        temp = geo[i][j]

        geo[i][j] = "_"

        for a in range(0, 4):

            idx, jdy = i + dx[a], j + dy[a]

            if isOnMap(idx, jdy):

                if isNeededToSolve(temp, idx, jdy):

                    search(idx, jdy)

    def isOnMap(i, j):
        return 0 <= i and 0 <= j and i < H and j < W

    def isNeededToSolve(temp, i, j):

        target = geo[i][j]

        return target is not "_" and temp is target

    limit = 10**7

    sys.setrecursionlimit(limit)

    H, W, tempH = -1, -1, 0

    dx = [-1, 0, 1, 0]

    dy = [0, -1, 0, 1]

    geo = [[0 for i in range(1)] for j in range(1)]

    while True:

        line = eval(input())

        if H is -1 and W is -1:

            p = line.split(" ")

            H, W = int(p[0]), int(p[1])

            geo = [[0 for i in range(W)] for j in range(H)]

        else:

            geo[tempH] = list(line)

            tempH += 1

        if H is 0 and W is 0:
            break

        if tempH is H:

            solve()

            H, W, tempH = -1, -1, 0


problem_p00118()
