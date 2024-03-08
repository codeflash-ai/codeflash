def problem_p00687():
    while 1:

        n, a, b = list(map(int, input().split()))

        if n == a == b == 0:
            break

        c = 0
        d = [1] + [0] * 2000000

        for i in range(0, n + 1):

            if d[i]:
                d[i + a] = d[i + b] = 1

            else:
                c += 1

        print(c)


problem_p00687()
