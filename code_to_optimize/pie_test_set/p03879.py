def problem_p03879():
    a, b, c, d, e, f = list(map(int, open(0).read().split()))

    a, b, c = sorted(
        (s * s + t * t) ** 0.5 for s, t in ((a - c, b - d), (c - e, d - f), (e - a, f - b))
    )

    d = a + b + c

    s = d / 2

    s = (s * (s - a) * (s - b) * (s - c)) ** 0.5

    r = 2 * s / d

    l, h = 0, 1e5

    while h - l > 1e-9:

        m = (h + l) / 2

        if c * (r - m) / r < 2 * m:
            h = m

        else:
            l = m

    print(l)


problem_p03879()
