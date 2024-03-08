def problem_p00554():
    n, m = list(map(int, input().split()))

    p = [list(map(int, input().split())) for i in range(m)]

    c = []

    for i in p:

        a, b = i

        if a >= n:
            o = 0

        else:
            o = n - a

        c.append(o)

    c.sort()

    c.pop()

    print((sum(c)))


problem_p00554()
