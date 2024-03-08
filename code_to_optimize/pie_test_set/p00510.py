def problem_p00510():
    # -*- coding: utf-8 -*-

    n = int(eval(input()))

    m = int(eval(input()))

    ans = c = m

    for i in range(n):

        a, b = [int(a) for a in input().split()]

        if c < 0:

            continue

        c += a - b

        if ans < c:

            ans = c

        if c < 0:

            ans = 0

    print(ans)


problem_p00510()
