def problem_p03731():
    n, t = list(map(int, input().split()))

    l = list(map(int, input().split()))

    l.append(l[-1] + t)

    s = 0

    for i in range(len(l) - 1):

        s += min(l[i + 1] - l[i], t)

    print(s)


problem_p03731()
