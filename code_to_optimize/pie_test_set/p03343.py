def problem_p03343():
    n, k, q, *a = list(map(int, open(0).read().split()))

    s = sorted

    m = 1e9

    for b in a:

        i = [i for i, t in enumerate(a) if t < b]

        l = s(sum([s(a[i + 1 : j])[::-1][k - 1 :] for i, j in zip([-1] + i, i + [n])], []))

        if len(l) >= q:
            m = min(m, l[q - 1] - b)

    print(m)


problem_p03343()
