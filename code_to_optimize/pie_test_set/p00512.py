def problem_p00512():
    n = int(eval(input()))

    o = {}

    for i in range(n):

        p, m = input().split()

        o[p] = o.get(p, 0) + int(m)

    k = list(o.keys())

    k = [(len(x), x) for x in k]

    k.sort()

    for i in k:

        print((i[1], o[i[1]]))


problem_p00512()
