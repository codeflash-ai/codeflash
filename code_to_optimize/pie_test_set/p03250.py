def problem_p03250():
    # インポート

    import numpy as np

    # A,B,Cの値の獲得

    num = list(map(int, input().split()))

    # numリストの最大値を10倍

    num[num.index(max(num))] = max(num) * 10

    # npで要素の和を計算し、出力

    print((np.sum(num)))


problem_p03250()
