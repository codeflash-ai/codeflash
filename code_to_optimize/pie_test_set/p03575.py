def problem_p03575():
    icase = 0

    if icase == 0:

        n, m = list(map(int, input().split()))

        a = []

        b = []

        for i in range(m):

            ai, bi = list(map(int, input().split()))

            a.append(ai)

            b.append(bi)

    elif icase == 1:

        n = 6

        m = 5

        #    a=[[1],[2],[3],[4],[5]]

        #    b=[[2],[3],[4],[5],[6]]

        a = [1, 2, 3, 4, 5]

        b = [2, 3, 4, 5, 6]

    elif icase == 2:

        n = 7

        m = 7

        a = [1, 2, 3, 4, 4, 5, 6]

        b = [3, 7, 4, 5, 6, 6, 7]

    vtx = []

    for i in range(n):

        vtx.append([i + 1])

    icnt = 0

    for j in range(m):

        vtx1 = vtx.copy()

        a1 = a.copy()

        b1 = b.copy()

        del a1[j]

        del b1[j]

        for i in range(m - 1):

            for v1 in vtx1:

                if a1[i] in v1:

                    if not b1[i] in v1:

                        for v2 in vtx1:

                            if v1 != v2:

                                if b1[i] in v2:

                                    break

                        vtx1.remove(v1)

                        vtx1.remove(v2)

                        v3 = v1 + v2

                        vtx1.append(v3)

        #        print(a[i],b[i],len(vtx1),vtx1)

        #    print(len(vtx1),vtx1)

        if len(vtx1) != 1:

            icnt = icnt + 1

    print(icnt)


problem_p03575()
