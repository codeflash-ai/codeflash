def problem_p04006():
    n, x, *a = list(map(int, open(0).read().split()))

    m = 1e18

    for i in range(n):
        m = min(m, sum(a) + x * i)
        a = [min(t) for t in zip(a, a[-1:] + a)]

    print(m)


problem_p04006()
