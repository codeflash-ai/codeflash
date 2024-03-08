def problem_p00209():
    def main(piece, c):

        for h in range(n - m + 1):

            for w in range(n - m + 1):

                if f(piece, h, w):

                    break

        if c != 4:

            move(piece, c)

        else:

            if ans:

                print(ans[0], ans[1])

            else:

                print("NA")

    def f(piece, h, w):

        global ans

        coordinate = None

        for _h in range(m):

            for _w in range(m):

                if piece[_h][_w] != "-1":

                    if piece[_h][_w] == picture[h + _h][w + _w]:

                        if coordinate is None:

                            coordinate = (int(w + _w + 1), int(h + _h + 1))

                            if ans:

                                if coordinate[1] > ans[1]:

                                    return

                                elif coordinate[1] == ans[1]:

                                    if coordinate[0] > ans[0]:

                                        return

                    else:

                        return

        else:

            ans = coordinate

            return True

    def move(piece, c):

        _piece = []

        for w in range(m):

            lis = []

            for h in range(m):

                lis.append(piece[h][w])

            else:

                lis.reverse()

                _piece.append(lis)

        else:

            main(_piece, c + 1)

    while True:

        n, m = list(map(int, input().split()))

        if n == m == 0:
            break

        picture = [input().split() for i in range(n)]

        piece = [input().split() for i in range(m)]

        ans = None

        main(piece, 1)


problem_p00209()
