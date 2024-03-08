def problem_p03317():
    import math

    n, k = list(map(int, input().split()))

    a = list(map(int, input().split()))

    m = a.index(min(a))

    print((math.ceil((len(a) - 1) / (k - 1))))


problem_p03317()
