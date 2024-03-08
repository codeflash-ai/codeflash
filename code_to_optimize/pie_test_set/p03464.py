def problem_p03464():
    import sys

    K = int(sys.stdin.readline())

    A = list(map(int, sys.stdin.readline().split()))

    # うまい方法がない気がするため、二分探索で攻める

    l = 0

    r = 2 * 10**15

    ans1 = -1

    while l != r:

        n = (l + r) // 2

        res = n

        for a in A:

            res = res - res % a

        if res >= 2:

            r = n

            if res == 2:

                ans1 = n

        else:

            l = n + 1

    l = 0

    r = 2 * 10**15

    ans2 = -1

    while l != r:

        n = (l + r) // 2

        res = n

        for a in A:

            res = res - res % a

        if res <= 2:

            l = n + 1

            if res == 2:

                ans2 = n

        else:

            r = n

    if ans1 != -1:

        print((ans1, ans2))

        # print(n - n % A[0], A[0] * ((n - 1) // A[0] + 1) - 1)

    else:

        print((-1))


problem_p03464()
