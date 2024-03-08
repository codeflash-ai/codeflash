def problem_p00203():
    # -*- coding: utf-8 -*-

    """

    http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=0203



    """

    import sys

    from sys import stdin

    from collections import deque, defaultdict

    input = stdin.readline

    def solve(field):

        BLANK, OBSTACLE, JUMP = 0, 1, 2

        ans = 0  #  ??????????????°???????????°

        dy = [1, 1, 1]  # ?????????????????????????????????????§???????

        dx = [0, -1, 1]

        x_limit = len(field[0])

        y_limit = len(field)

        path = defaultdict(
            int
        )  #  ??????????????????????????°???????????°????????????'x???_y???'???????????????

        Q = deque()

        for x, m in enumerate(field[0]):

            if m == BLANK:  #  ?????????????????°?????´?????????????????????????????????

                t = "{}_{}".format(x, 0)

                Q.append((x, 0))

                path[t] = 1

        while Q:

            cx, cy = Q.popleft()  #  ?????¨??°?????§?¨?

            t = "{}_{}".format(cx, cy)

            num = path.pop(t)

            if field[cy][cx] == OBSTACLE:

                continue

            elif field[cy][cx] == JUMP:  #  ?????£????????§?????°?????????????????£????????°

                if cy + 2 > y_limit - 1:

                    ans += num

                else:

                    t = "{}_{}".format(cx, cy + 2)

                    if not path[t]:

                        Q.append((cx, cy + 2))

                    path[t] += num

                continue

            elif cy == y_limit - 1:

                ans += num

                continue

            for i in range(len(dx)):

                nx = cx + dx[i]  #  ?????°????????§?¨?

                ny = cy + dy[i]

                if 0 <= nx < x_limit:

                    if (ny >= y_limit - 1) and field[ny][nx] == BLANK:

                        ans += num

                    else:

                        if (
                            field[ny][nx] == JUMP and dx[i] == 0
                        ):  #  ?????£????????°????????£??????????????\????????´???

                            if ny + 2 > y_limit - 1:

                                ans += num

                            else:

                                t = "{}_{}".format(nx, ny + 2)

                                if not path[t]:

                                    Q.append((nx, ny + 2))

                                path[t] += num

                        elif field[ny][nx] == BLANK:

                            t = "{}_{}".format(nx, ny)

                            if not path[t]:

                                Q.append((nx, ny))

                            path[t] += num

        return ans

    def main(args):

        while True:

            X, Y = list(map(int, input().strip().split()))

            if X == 0 and Y == 0:

                break

            field = []

            for _ in range(Y):

                temp = [int(x) for x in input().strip().split()]

                field.append(temp)

            result = solve(field)

            print(result)

    if __name__ == "__main__":

        main(sys.argv[1:])


problem_p00203()
