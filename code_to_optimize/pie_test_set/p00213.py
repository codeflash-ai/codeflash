def problem_p00213():

    def putpiece(bitmap, unused, pieces, numans, pcsans, FINFLAG):

        # print(bin(bitmap))

        # print(unused,"***",pieces,"***",numans)

        if FINFLAG:

            return numans, pcsans

        if not unused:

            pcsans = pieces

            return numans + 1, pcsans

        if numans > 1:

            return 2, pieces

        b, k, y, x = unused[-1]

        # print(set([ (k//i,k//(k//i)) for i in range(1,min(X+1,k+1))]))

        for h, w in set(
            [
                (k // i, k // (k // i))
                for i in range(1, min(X + 1, k + 1))
                if not (k // i) * (k // (k // i)) - k
            ]
        ):

            for pt, qt in product(
                list(range(max(0, y - h + 1), min(Y - h + 1, y + 1))),
                list(range(max(0, x - w + 1), min(X - w + 1, x + 1))),
            ):

                rt, st = pt + h - 1, qt + w - 1

                piece = 0

                piece2 = 0

                for i in range(st - qt + 1):
                    piece |= 1 << i

                for j in range(rt - pt + 1):
                    piece2 |= piece << j * X

                piece = piece2

                piece = piece << X - st - 1

                piece = piece << (Y - rt - 1) * X

                mark = 1

                mark = mark << X - x - 1

                mark = mark << (Y - y - 1) * X

                if not (bitmap & piece) ^ mark:
                    numans, pcsans = putpiece(
                        bitmap | piece,
                        unused[:-1],
                        pieces + [[b, k, pt, qt, rt, st]],
                        numans,
                        pcsans,
                        False,
                    )

                if numans > 1:
                    return 2, pcsans

            if numans > 1:
                return 2, pcsans

        else:

            return numans, pcsans

    from itertools import product

    while True:

        numans = 0

        pcsans = []

        X, Y, n = list(map(int, input().split()))

        if not X:
            break

        bk = sorted([list(map(int, input().split())) for _ in range(n)])

        ss = [list(map(int, input().split())) for _ in range(Y)]

        yxs = sorted(
            [ss[i][j], i, j] for i, j in product(list(range(Y)), list(range(X))) if ss[i][j]
        )

        bkyx = [bk[i] + yxs[i][1:] for i in range(n)]

        # from pprint import pprint

        # pprint(bk)

        # pprint(ss)

        filled = "".join(["".join(["1" if ss[i][j] else "0" for j in range(X)]) for i in range(Y)])

        # pprint(filled)

        # pprint([filled[i*X:i*X+X] for i in range(Y)])

        # pprint(bkyx)

        nans, pcs = putpiece(int(filled, 2), bkyx, [], 0, 0, False)

        # print(nans,pcs)

        if nans > 1:

            print("NA")

        elif nans:

            toprint = [[0] * X for _ in range(Y)]

            for i, m, sy, sx, ey, ex in pcs:

                for j in range(sy, ey + 1):
                    toprint[j][sx : ex + 1] = [i] * (ex - sx + 1)

            for a in toprint:
                print((" ".join(str(b) for b in a)))

        else:

            print("NA")

        # print("***********************************::")

        # break

    # numans = 0

    # mapans = []

    # bitmap,[[b1,k1,x1,y1]],[[p,q,r,s]]

    # if not len([]):

    #  numans += 0

    #  mapans = p,q,r,s

    #  return None

    # pt,qt,rt,st

    # if not bitmap & bitpiece

    #  bitmap |= bitpeace

    #  putpiece(bitmap|bitpiece,[][:-1],[]+[pt,qt,rt,st])


problem_p00213()
