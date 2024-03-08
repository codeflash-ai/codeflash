def problem_p02935():
    n = int(eval(input()))

    v = list(map(int, input().split()))

    v = sorted(v)

    p = v[0]

    for i in range(1, n):

        p = (v[i] + p) / 2

    print(p)


problem_p02935()
