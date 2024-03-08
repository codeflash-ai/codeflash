def problem_p00210():
    from sys import stdin

    from itertools import chain

    head = [[8, 1, 2, 4], [1, 2, 4, 8], [2, 4, 8, 1], [4, 8, 1, 2]]

    while True:

        W, H = list(map(int, stdin.readline().split()))

        m = [[0] * W for _ in range(H)]

        ps = []

        if not (W or H):
            break

        flag = True

        for h in range(H):

            s = stdin.readline()

            if "X" in s:
                flag = False

            for w in range(W):

                if s[w] in [".", "X", "E", "N", "W", "S"]:

                    m[h][max(0, w - 1)] |= 1

                    m[min(H - 1, h + 1)][w] |= 2

                    m[h][min(W - 1, w + 1)] |= 4

                    m[max(0, h - 1)][w] |= 8

                if s[w] in ["E", "N", "W", "S"]:

                    ps.append([h, w, ["E", "N", "W", "S"].index(s[w])])

                if s[w] == "X":

                    m[h][w] |= 16

        # if flag:

        #     print("NA")

        #     continue

        if W <= 2 or H <= 2:

            print((1))

            continue

        # print(ps)

        # for n in m:

        #     print(n)

        ttt = 0

        while True:

            ttt += 1

            if ttt > 180:
                print("NA")
                break

            # import time

            # time.sleep(1)

            mt = [[0] * W for _ in range(H)]

            for p in ps:

                mt[p[0]][p[1]] |= 1

            dest = []

            # for mmm in mt:

            #     print(mmm)

            # print()

            # print([a[:2] for a in ps])

            for pi, p in enumerate(ps):

                for i in range(4):

                    if head[p[2]][i] & m[p[0]][p[1]]:

                        d = head[p[2]][i]

                        if d == 1 and not [p[0], p[1] + 1] in [a[:2] for a in ps]:
                            p[2] = 0
                            dest.append([pi, 2, p[0], p[1] + 1])
                            break

                        elif d == 2 and not [p[0] - 1, p[1]] in [a[:2] for a in ps]:
                            p[2] = 1
                            dest.append([pi, 3, p[0] - 1, p[1]])
                            break

                        elif d == 4 and not [p[0], p[1] - 1] in [a[:2] for a in ps]:
                            p[2] = 2
                            dest.append([pi, 0, p[0], p[1] - 1])
                            break

                        elif d == 8 and not [p[0] + 1, p[1]] in [a[:2] for a in ps]:
                            p[2] = 3
                            dest.append([pi, 1, p[0] + 1, p[1]])
                            break

                else:

                    dest.append([pi, (p[2] + 2) & 3, p[0], p[1]])

            dest = sorted(dest, key=lambda x: (x[2:], x[1]))

            # print(dest)

            # for mmm in mt:

            #     print(mmm)

            # print("ps = ",ps,dest)

            dellist = []

            for pi, d, dy, dx in dest:

                # print(W,H,dy,dx,mt)

                if not mt[dy][dx]:

                    # print("move",W,H,ps[pi][:2],dy,dx)

                    mt[dy][dx] |= 1

                    ps[pi][:2] = [dy, dx]

                if m[ps[pi][0]][ps[pi][1]] & 16:

                    # print("OUT")

                    # print(ttt)

                    # time.sleep(3)

                    dellist.append(pi)

            # print(dellist)

            for idx in sorted(dellist)[::-1]:

                del ps[idx]

            if not len(ps):

                print(ttt)

                break

        # ms = [stdin.readline() for _ in range(H)]

        # print(m)

        # from pprint import pprint

        # print(ps)

        # for n in m:

        #     print(n)

        # print()


problem_p00210()
