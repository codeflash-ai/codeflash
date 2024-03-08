def problem_p03472():
    n, h = list(map(int, input().split()))

    a, b = [], []

    for _ in range(n):

        ai, bi = list(map(int, input().split()))

        a.append(ai), b.append(bi)

    a.sort(), b.sort()

    ans, amax = 0, a[-1]

    for bi in b[::-1]:

        if bi <= amax or h <= 0:

            break

        h -= bi

        ans += 1

    print((ans + ((h + amax - 1) // amax) * (h > 0)))


problem_p03472()
