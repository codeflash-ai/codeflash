def problem_p02683():
    import numpy as np

    n, m, x = list(map(int, input().split()))

    ca = [list(map(int, input().split())) for _ in range(n)]

    a = [np.array(i[1:]) for i in ca]

    c = [i[0] for i in ca]

    xx = np.array([x] * m)

    ans = pow(10, 9)

    for i in range(2**n):

        count = 0

        aa = np.array([0] * m)

        for j in range(n):

            if (i >> j) & 1:

                aa += a[j]

                count += c[j]

            if all(i >= x for i in aa):

                ans = min(ans, count)

    print((ans if ans < pow(10, 9) else -1))


problem_p02683()
