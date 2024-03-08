def problem_p02235():
    from bisect import bisect_left

    q = int(eval(input()))

    for _ in [0] * q:

        s1, s2 = eval(input()), eval(input())

        l1, l2 = len(s1), len(s2)

        p = []

        for i, c in enumerate(s1):

            s = 0

            while True:

                j = s2[s:].find(c)

                if j == -1:

                    break

                p.append((i, s + j))

                s += j + 1

        lis = []

        for _, y in sorted(p, key=lambda x: (x[0], -x[1])):

            i = bisect_left(lis, y)

            if len(lis) <= i:

                lis.append(y)

            else:

                lis[i] = y

        print((len(lis)))


problem_p02235()
