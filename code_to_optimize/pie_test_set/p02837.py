def problem_p02837():
    def popcount(x):

        r = 0

        while x:

            if x & 1:

                r += 1

            x >>= 1

        return r

    n = int(eval(input()))

    r = list(range(n))

    a = [[0, 0] for _ in r]

    for i in r:

        for _ in range(int(eval(input()))):

            x, y = list(map(int, input().split()))

            a[i][y] |= 1 << (x - 1)

    m = 0

    l = (1 << n) - 1

    for x in range(1, l + 1):

        if all(a[i][0] & x == a[i][1] & l - x == 0 for i in r if x >> i & 1):

            m = max(m, popcount(x))

    print(m)


problem_p02837()
