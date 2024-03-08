def problem_p02689():
    # -*- coding: utf-8 -*-

    # 標準入力の取得

    N, M = list(map(int, input().split()))

    H = list(map(int, input().split()))

    A, B = [], []

    for m in range(M):

        A_m, B_m = list(map(int, input().split()))

        A.append(A_m)

        B.append(B_m)

    # 求解処理

    observatory = [True for n in range(N)]

    for A_m, B_m in zip(A, B):

        lower_observatory = int()

        if H[A_m - 1] < H[B_m - 1]:

            observatory[A_m - 1] = False

        elif H[A_m - 1] > H[B_m - 1]:

            observatory[B_m - 1] = False

        else:

            observatory[A_m - 1] = False

            observatory[B_m - 1] = False

    result = sum(observatory)

    # 結果出力

    print(result)


problem_p02689()
