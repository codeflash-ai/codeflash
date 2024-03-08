def problem_p02258():
    N = int(eval(input()))

    datas = [int(eval(input())) for _ in range(N)]

    now_max = max(datas[1:])

    max_index = (N - 1) - datas[::-1].index(now_max)

    _min = min(datas)

    min_index = datas.index(_min)

    if max_index > min_index:

        diff = now_max - _min

    else:

        diff = now_max - datas[0]

        for i in range(1, N - 1):

            if i == max_index:

                now_max = max(datas[i + 1 :])

                max_index = datas.index(now_max, i + 1)

            new_diff = now_max - datas[i]

            if diff < new_diff:

                diff = new_diff

    print(diff)


problem_p02258()
