def problem_p02748():
    a, b, m = list(map(int, input().split()))

    dr = list(map(int, input().split()))

    wa = list(map(int, input().split()))

    xyc = [list(map(int, input().split())) for i in range(m)]

    low = 200001

    for i in range(m):

        f = xyc[i][0] - 1

        d = xyc[i][1] - 1

        if low >= dr[f] + wa[d] - xyc[i][2]:

            low = dr[f] + wa[d] - xyc[i][2]

    dr.sort()

    wa.sort()

    if low >= (dr[0] + wa[0]):

        print((dr[0] + wa[0]))

    else:

        print(low)


problem_p02748()
