def problem_p03661():
    import itertools

    N = int(eval(input()))

    A = [int(_) for _ in input().split()]

    ts = list(itertools.accumulate(A))

    total = ts[-1]

    total_half = total / 2

    r = sorted(ts[:-1], key=lambda x: abs(x - total_half))[0]

    print((abs(total - r - r)))


problem_p03661()
