def problem_p03361():
    H, W = list(map(int, input().split()))

    s = [list(eval(input())) for i in range(H)]

    dh = [-1, 0, 0, 1]

    dw = [0, -1, 1, 0]

    cnt = 0

    for h in range(H):

        for w in range(W):

            # Check only black cell

            if s[h][w] == ".":

                continue

            for k in range(4):

                nh = h + dh[k]

                nw = w + dw[k]

                if nh < 0 or nw < 0 or H <= nh or W <= nw:

                    continue

                # ok

                if s[nh][nw] == "#":

                    break

            # failed

            else:

                cnt += 1

    if cnt == 0:

        print("Yes")

    else:

        print("No")


problem_p03361()
