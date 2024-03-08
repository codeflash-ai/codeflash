def problem_p00074():
    N = 60

    def f(t):

        c, t = t % N, t / N

        b, a = t % N, t / N

        print("%02d:%02d:%02d" % (a, b, c))

        return

    t1 = 2 * N * N

    while 1:

        h, m, s = list(map(int, input().split()))

        if [h, m, s] == [-1, -1, -1]:
            break

        t2 = (h * N + m) * N + s

        t = t1 - t2

        f(t)

        f(t * 3)


problem_p00074()
