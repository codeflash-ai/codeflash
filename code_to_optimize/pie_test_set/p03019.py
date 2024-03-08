def problem_p03019():
    N, X = list(map(int, input().split()))

    items = []

    for i in range(N):

        b, l, u = list(map(int, input().split()))

        items.append((b, l, u, u * (X - b), -b * l))

    items = sorted(items, key=lambda x: -(x[3] - x[4]))

    def f(num):

        cnt = num // X

        mod = num % X

        point = 0

        for i in range(cnt):

            point += items[i][3]

        for i in range(cnt, N):

            point += items[i][4]

        if mod == 0:

            if point >= 0:

                return True

            else:

                return False

        else:

            tmp = -float("inf")

            for i, tmp2 in enumerate(items):

                b, l, u, d1, d2 = tmp2

                if mod >= b:

                    mul = u * (mod - b)

                else:

                    mul = l * (mod - b)

                if i < cnt:

                    tmp = max(tmp, point - d1 + mul + items[cnt][3] - items[cnt][4])

                else:

                    tmp = max(tmp, point - d2 + mul)

            if tmp >= 0:

                return True

            else:

                return False

    ok = N * X + 1

    ng = -1

    while (ok - ng) > 1:

        mid = (ok + ng) // 2

        if f(mid):

            ok = mid

        else:

            ng = mid

    print(ok)


problem_p03019()
