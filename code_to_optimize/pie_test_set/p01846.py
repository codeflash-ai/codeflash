def problem_p01846():

    while True:

        s = eval(input())

        if s == "#":

            break

        rows = s.split("/")

        a, b, c, d = list(map(int, input().split()))

        a -= 1

        b -= 1

        c -= 1

        d -= 1

        grid = ["" for _ in rows]

        for i, row in enumerate(rows):

            for kk in row:

                if kk == "b":

                    grid[i] += "b"

                else:

                    grid[i] += "." * int(kk)

        grid[a] = grid[a][:b] + "." + grid[a][b + 1 :]

        grid[c] = grid[c][:d] + "b" + grid[c][d + 1 :]

        ans = []

        # for row in grid:

        #     print(row)

        for row in grid:

            s = ""

            tmp = 0

            for kk in row:

                if kk == "b":

                    if tmp != 0:

                        s += str(tmp)

                        tmp = 0

                    s += "b"

                else:

                    tmp += 1

            if tmp != 0:

                s += str(tmp)

                tmp = 0

            ans.append(s)

        print(("/".join(ans)))


problem_p01846()
