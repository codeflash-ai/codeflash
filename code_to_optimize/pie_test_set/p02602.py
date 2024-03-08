def problem_p02602():
    # -*- coding: utf-8 -*-

    # 標準入力を取得

    N, K = list(map(int, input().split()))

    A = list(map(int, input().split()))

    # 求解処理

    for i in range(K, N):

        if A[i] > A[i - K]:

            print("Yes")

        else:

            print("No")


problem_p02602()
