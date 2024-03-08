def problem_p02727():
    x, y, a, b, c = list(map(int, input().split()))

    p = list([["red", int(x)] for x in input().split()])

    q = list([["green", int(x)] for x in input().split()])

    r = list([["nan", int(x)] for x in input().split()])

    p.sort(key=lambda x: x[1])

    p = p[-x:]

    q.sort(key=lambda x: x[1])

    q = q[-y:]

    p[len(p) : len(p)] = q

    p[len(p) : len(p)] = r

    p.sort(key=lambda x: x[1])

    p = p[-x - y :]

    print((sum([pi[1] for pi in p])))


problem_p02727()
