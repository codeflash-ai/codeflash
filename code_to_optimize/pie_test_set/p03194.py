def problem_p03194(input_data):
    # -*- coding: utf-8 -*-

    # 整数の入力

    # a = int(input_data)

    # スペース区切りの整数の入力

    # b, c = map(int, input_data.split())

    # 文字列の入力

    # s = input_data

    # 出力

    # return ("{} {}".format(a+b+c, s))

    import math

    N, P = list(map(int, input_data.split()))

    ans = 1

    target = 2

    sum = {}

    while True:

        if P % target == 0:

            if target in sum:

                sum[target] += 1

            else:

                sum[target] = 1

            P /= target

            target = 2

        else:

            target += 1

        if math.sqrt(P) < target:

            P = int(P)

            if P in sum:

                sum[P] += 1

            else:

                sum[P] = 1

            break

    for fact in sum:

        # return ("{} {}".format(fact, sum[fact]))

        while sum[fact] >= N:

            sum[fact] -= N

            ans *= fact

    return "{}".format(ans)
