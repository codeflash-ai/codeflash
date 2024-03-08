def problem_p02943():
    n, k = list(map(int, input().split()))

    s = eval(input())

    r = list(range(n))

    u = min((s + s[::-1])[i:] for i in r)

    i = min(i for i in r if u[i] != u[0])

    print(((u[0] * min(i << k - 1, n) + u[i:])[:n]))


problem_p02943()
