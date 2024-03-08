def problem_p03645():
    # -*- coding: utf-8 -*-

    N, M = list(map(int, input().split()))

    abM = [list(map(int, input().split())) for i in range(M)]

    transit = []

    transit2 = []

    for i in range(M):

        if abM[i][0] == 1:

            transit.append(abM[i][1])

        if abM[i][1] == N:

            transit2.append(abM[i][0])

    # setで共通要素の集合を取ってみる

    if len(set(transit) & set(transit2)) != 0:

        print("POSSIBLE")

    else:

        print("IMPOSSIBLE")


problem_p03645()
