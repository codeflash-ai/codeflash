def problem_p02300():
    def isCLKWISE(ph):

        return not (
            (ph[-1][0] - ph[-3][0]) * (-ph[-3][1] + ph[-2][1])
            - (ph[-2][0] - ph[-3][0]) * (-ph[-3][1] + ph[-1][1])
            < 0
        )

    def ConvexHullScan(P):

        P.sort()

        phU = [P[0], P[1]]

        for p in P[2:]:

            phU.append(p)

            while True:

                if isCLKWISE(phU):
                    break

                else:

                    del phU[-2]

                    if len(phU) == 2:
                        break

        phL = [P[-1], P[-2]]

        for p in P[-3::-1]:

            phL.append(p)

            while True:

                if isCLKWISE(phL):
                    break

                else:

                    del phL[-2]

                    if len(phL) == 2:
                        break

        ph = phU + phL[1:-1]

        return ph

    n = list(range(int(eval(input()))))

    P = []

    for i in n:

        P.append([int(x) for x in input().split()])

    Q = ConvexHullScan(P)

    Q.reverse()

    print((len(Q)))

    idx = min([[x[1][1], x[1][0], x[0]] for x in enumerate(Q)])[2]

    R = Q[idx:] + Q[:idx]

    for r in R:
        print((r[0], r[1]))


problem_p02300()
