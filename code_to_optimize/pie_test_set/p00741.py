def problem_p00741():
    import sys

    def visit(x, y, area):

        area[y][x] = 0

        move = [(-1, 0), (0, 1), (1, 0), (0, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]

        # print '(' + str(x) + ',' + str(y) + ')'

        for i in move:

            if 0 <= (x + i[0]) < w and 0 <= (y + i[1]) < h and area[y + i[1]][x + i[0]] == 1:

                visit(x + i[0], y + i[1], area)

    def solve():

        area = []

        ans = 0

        for i in range(h):

            area.append(list(map(int, input().split())))

        # print 'w: ' + str(w) + ',h: ' + str(h)

        for i in range(h):

            for j in range(w):

                if area[i][j] == 1:

                    ans += 1

                    visit(j, i, area)

        print(ans)

    if __name__ == "__main__":

        sys.setrecursionlimit(100000)

        while True:

            w, h = 0, 0

            w, h = list(map(int, input().split()))

            if w == 0 and h == 0:

                break

            solve()


problem_p00741()
