def problem_p03617():
    q, h, s, d = list(map(int, input().split()))

    n = int(eval(input()))

    twoL = min(q * 8, 4 * h, 2 * s, d)

    oneL = min(q * 4, 2 * h, s)

    p = (n // 2) * twoL + (n % 2) * oneL

    print(p)


problem_p03617()
