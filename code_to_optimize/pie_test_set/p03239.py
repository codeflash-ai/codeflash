def problem_p03239():
    n, t = list(map(int, input().split()))
    c = []

    for i in range(n):

        a, b = list(map(int, input().split()))

        if b <= t:

            c.append(a)

    if c == []:
        print("TLE")

    else:
        c.sort()
        print((c[0]))


problem_p03239()
