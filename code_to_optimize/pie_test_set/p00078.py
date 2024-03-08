def problem_p00078():
    def magic(n):

        c = 1

        x = n / 2

        y = x + 1

        while 1:

            A[y][x] = c

            if c == n * n:
                break

            while 1:

                x, y = (x + 1) % n, (y + 1) % n

                if A[y][x] == 0:
                    break

                x, y = (x - 1) % n, (y + 1) % n

                if A[y][x] == 0:
                    break

            c += 1

        return

    while 1:

        n = eval(input())

        if n == 0:
            break

        N = list(range(n))

        A = [[0] * n for i in N]

        magic(n)

        for i in N:

            print("".join(["%4d" % (e) for e in A[i]]))


problem_p00078()
