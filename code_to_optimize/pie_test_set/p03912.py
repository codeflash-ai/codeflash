def problem_p03912():
    from collections import defaultdict as dd

    N, M = list(map(int, input().split()))

    c = list(map(int, input().split()))

    c.sort()

    m = dd(int)

    d = dd(int)

    for n in c:

        d[n] += 1

        m[n % M] += 1

    res = 0

    for i in range(M):

        temp = min(m[i], m[(M - i) % M])

        if i == (M - i) % M:

            temp = m[i] // 2

        m[i] -= temp

        m[(M - i) % M] -= temp

        res += temp

    for i in range(N):

        n = c[i]

        if m[n % M] > 1:

            k = m[n % M] // 2

            if d[n] > 1:

                temp = min(d[n] // 2, k)

                res += temp

                d[n] -= temp * 2

                m[n % M] -= temp * 2

    print(res)


problem_p03912()
