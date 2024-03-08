def problem_p03225():
    H, W = list(map(int, input().split()))

    S = [list(eval(input())) for i in range(H)]

    table = [[0] * (H + W - 1) for i in range(H + W - 1)]

    for j in range(H):

        for i in range(W):

            if S[j][i] == "#":

                table[i + j][i - j + H - 1] = 1

    yoko = [[0] * (H + W) for i in range(H + W - 1)]

    for j in range(H + W - 1):

        for i in range(1, H + W):

            yoko[j][i] = yoko[j][i - 1] + table[j][i - 1]

    tate = [[0] * (H + W - 1) for i in range(H + W)]

    for j in range(1, H + W):

        for i in range(H + W - 1):

            tate[j][i] = tate[j - 1][i] + table[j - 1][i]

    ans = 0

    for y in range(H + W - 1):

        for x in range((y + H - 1) % 2, H + W - 1, 2):

            if table[y][x] != 1:

                continue

            for z in range(x + 2, H + W - 1, 2):

                if table[y][z] == 1:

                    d = z - x

                    if y + d < H + W - 1:

                        ans += yoko[y + d][z + 1] - yoko[y + d][x]

                        # print(1,'.',ans,':',x,y,z)

                    if y - d >= 0:

                        ans += yoko[y - d][z + 1] - yoko[y - d][x]

                        # print(2,'.',ans,':',x,y,z)

            for w in range(y + 2, H + W - 1, 2):

                if table[w][x] == 1:

                    e = w - y

                    if x + e < H + W - 1:

                        ans += tate[w][x + e] - tate[y + 1][x + e]

                        # print(3,'.',ans,':',x,y,w)

                    if x - e >= 0:

                        ans += tate[w][x - e] - tate[y + 1][x - e]

                        # print(4,'.',ans,':',x,y,w)

    print(ans)


problem_p03225()
