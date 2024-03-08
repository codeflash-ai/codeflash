def problem_p00592():
    def cv(t):

        return t / 100 * 60 + t % 100

    while 1:

        n, p, q = list(map(int, input().split()))

        if n == 0:
            break

        v = [0] * 1440

        for i in range(n):

            k = int(input())

            a = list(map(int, input().split()))

            for j in range(k):

                for l in range(cv(a[::2][j]), cv(a[1::2][j])):

                    v[l] += 1

        m = c = 0

        for i in range(cv(p), cv(q)):

            c = c + 1 if v[i] < n else 0

            m = max(m, c)

        print(m)


problem_p00592()
