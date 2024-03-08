def problem_p00085():
    while 1:

        n, m = list(map(int, input().split()))

        if n == 0:
            break

        a = m - 1

        while a < m * n - n:
            a = m * a // (m - 1) + 1

        print((n * m - a))


problem_p00085()
