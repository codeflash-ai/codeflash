def problem_p02625():
    n, m = list(map(int, input().split()))

    a, M, d = 1, 10**9 + 7, [1]

    for i in range(1, n + 1):

        d.append(((m - n + i - 1) * d[i - 1] + (i - 1) * d[i - 2]) % M)

        a = a * (m - i + 1) % M

    print((a * d[-1] % M))


problem_p02625()
