def problem_p01525():
    from collections import defaultdict

    MAX = 3652425

    n, q = list(map(int, input().split()))

    lst = [tuple(map(int, input().split())) for _ in range(n)]

    restor = [0] * (MAX + 10010)

    t0s = [0] * (MAX + 10010)

    t1s = [0] * (MAX + 10010)

    t2s = [0] * (MAX + 10010)

    t3s = [0] * (MAX + 10010)

    t1_cnt_save = defaultdict(int)

    t3_cnt_save = defaultdict(int)

    t1_cnt = 0

    t3_cnt = 0

    index = 0

    for i, line in enumerate(lst):

        w, t, x = line

        while index < MAX and w > restor[index]:

            t0s[index + 1] += t0s[index]

            t1_cnt -= t1_cnt_save[index + 1]

            t1s[index + 1] += t1s[index] + t1_cnt

            t3_cnt -= t3_cnt_save[index + 1]

            t3s[index + 1] += t3s[index] + 2 * t3_cnt

            t2s[index + 1] += t2s[index] + t3s[index + 1]

            restor[index + 1] = restor[index] + 1 + t0s[index] + t1s[index] + t2s[index]

            index += 1

        if w <= restor[index]:

            print(index)

            if t == 0:

                t0s[index] += 1

                t0s[index + x] -= 1

            elif t == 1:

                t1_cnt += 1

                t1_cnt_save[index + x] += 1

                t1s[index] += 1

                t1s[index + x] -= x

            elif t == 2:

                t3_cnt += 1

                t3_cnt_save[index + x] += 1

                t3s[index] += 1

                t3s[index + x] -= x * 2 - 1

                t2s[index] += 1

                t2s[index + x] -= x**2

        else:
            print("Many years later")

    for _ in range(q):

        y = int(eval(input()))

        while index < y:

            t0s[index + 1] += t0s[index]

            t1_cnt -= t1_cnt_save[index + 1]

            t1s[index + 1] += t1s[index] + t1_cnt

            t3_cnt -= t3_cnt_save[index + 1]

            t3s[index + 1] += t3s[index] + 2 * t3_cnt

            t2s[index + 1] += t2s[index] + t3s[index + 1]

            restor[index + 1] = restor[index] + 1 + t0s[index] + t1s[index] + t2s[index]

            index += 1

        print((restor[y]))


problem_p01525()
