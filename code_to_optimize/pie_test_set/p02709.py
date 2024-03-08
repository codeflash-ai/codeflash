def problem_p02709():
    e = enumerate

    n, a = open(0)

    n = int(n)

    d = [0] + [1e18] * n

    for j, (a, i) in e(sorted((-int(a), i) for i, a in e(a.split()))):
        d = [min(t + a * abs(~i - j + k + n), d[k - 1] + a * abs(~i + k)) for k, t in e(d)]

    print((-min(d)))


problem_p02709()
