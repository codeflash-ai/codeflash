def problem_p00486():
    from bisect import bisect_left as bl

    from bisect import bisect_right as br

    INF = 10**20

    w, h = list(map(int, input().split()))

    n = int(eval(input()))

    xlst = []

    ylst = []

    for i in range(n):

        x, y = list(map(int, input().split()))

        xlst.append(x)

        ylst.append(y)

    sorted_xlst = sorted(xlst)

    sorted_xlst_d = sorted(xlst * 2)

    sorted_ylst = sorted(ylst)

    sorted_ylst_d = sorted(ylst * 2)

    accx = accy = 0

    cum_sum_xlst = []

    cum_sum_ylst = []

    for i in range(n):

        accx += sorted_xlst[i]

        accy += sorted_ylst[i]

        cum_sum_xlst.append(accx)

        cum_sum_ylst.append(accy)

    clx = sorted_xlst_d[n - 1]

    crx = sorted_xlst_d[n]

    cly = sorted_ylst_d[n - 1]

    cry = sorted_ylst_d[n]

    num = n * 2 - 1

    ans = INF

    ansx = 10**10

    ansy = 10**10

    for i in range(n):

        xi = xlst[i]

        yi = ylst[i]

        if xi <= clx:

            cx = crx

        else:

            cx = clx

        if yi <= cly:

            cy = cry

        else:

            cy = cly

        px = bl(sorted_xlst, cx)

        py = bl(sorted_ylst, cy)

        if px:

            xlen = (
                (cx * px - cum_sum_xlst[px - 1]) * 2
                + (accx - cum_sum_xlst[px - 1] - cx * (n - px)) * 2
                - abs(xi - cx)
            )

        else:

            xlen = (accx - cx * n) * 2 - abs(xi - cx)

        if py:

            ylen = (
                (cy * py - cum_sum_ylst[py - 1]) * 2
                + (accy - cum_sum_ylst[py - 1] - cy * (n - py)) * 2
                - abs(yi - cy)
            )

        else:

            ylen = (accy - cy * n) * 2 - abs(yi - cy)

        tlen = xlen + ylen

        if ans > tlen:

            ans = tlen

            ansx = cx

            ansy = cy

        elif ans == tlen:

            if ansx > cx:

                ansx = cx

                ansy = cy

            elif ansx == cx:

                if ansy > cy:

                    ansy = cy

    print(ans)

    print((ansx, ansy))


problem_p00486()
