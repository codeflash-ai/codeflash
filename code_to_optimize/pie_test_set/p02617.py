def problem_p02617():
    n = int(eval(input()))

    d = [0] * n

    present = [[] for i in range(n)]

    for i in range(n - 1):

        u, v = list(map(int, input().split()))

        u -= 1

        v -= 1

        if u > v:

            u, v = v, u

        d[v] += 1

        present[u].append(v)

    ans = 0

    cur = 0

    for v in range(n):

        cur += (n - v) * d[v]

    for l in range(n):

        ans += (l + 1) * (l + 2) // 2

        ans -= cur

        for v in present[l]:

            cur -= n - v

    print(ans)


problem_p02617()
