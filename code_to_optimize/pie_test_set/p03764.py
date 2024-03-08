def problem_p03764():
    ai = lambda: list(map(int, input().split()))

    n, m = ai()

    x = ai()

    y = ai()

    mod = 10**9 + 7

    ans = 0

    sx, ssx, sy, ssy = [x[0]], [x[-1]], [y[0]], [y[-1]]

    for i in range(1, n):

        sx.append(sx[-1] + x[i])

        ssx.append(ssx[-1] + x[-i - 1])

    for i in range(1, m):

        sy.append(sy[-1] + y[i])

        ssy.append(ssy[-1] + y[-i - 1])

    a = sum(ssx[i] - sx[i] for i in range(n)) % mod

    b = sum(ssy[i] - sy[i] for i in range(m)) % mod

    ans = a * b % mod

    print(ans)


problem_p03764()
