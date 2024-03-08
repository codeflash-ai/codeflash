def problem_p00855():
    M = 1299710

    isPrime = [1] * M

    isPrime[0], isPrime[1] = 0, 0

    lp = [0] * M
    p = []

    b = 2

    for i in range(2, M):

        if isPrime[i]:

            for j in range(i * i, M, i):

                isPrime[j] = 0

            p.append(i)

            b = i

        lp[i] = b

    while 1:

        n = eval(input())

        if n == 0:
            break

        print(p[p.index(lp[n]) + 1] - lp[n] if lp[n] != n else "0")


problem_p00855()
