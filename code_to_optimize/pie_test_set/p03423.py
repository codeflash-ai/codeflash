def problem_p03423(input_data):
    # A - Grouping 2

    # ある学校に N 人の生徒がいる

    # なるべく多く 3人 以上のグループにしたい！

    # 3人以上のグループを 最大で x 個 作れるときの x の値を出力する

    # 生徒数 N （整数） を入力

    N = int(eval(input_data))

    # return (N)

    # N を 3 で割った余りで 計算する

    if N % 3 == 0:

        # return ('amari0')

        answer = int(N / 3)

    elif N % 3 == 1:

        # return ('amari1')

        answer = int((N - 1) / 3)

    else:

        # return ('amari2')

        answer = int((N - 2) / 3)

    # 結果の表示

    return answer
