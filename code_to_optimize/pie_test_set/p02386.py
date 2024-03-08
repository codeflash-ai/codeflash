def problem_p02386():
    n = int(eval(input()))

    a = [list(map(int, input().split())) for _ in range(n)]

    def f():

        for i in range(n - 1):

            d = a[i][:]
            d[3], d[4] = d[4], d[3]

            for j in range(i + 1, n):

                e = a[j][:]
                e[3], e[4] = e[4], e[3]

                for p in ("012345", "152043", "215304", "302541", "410352", "514320"):

                    f = [d[int(k)] for k in p]

                    g = f[1:5] * 2

                    for k in range(4):

                        if (g[k : k + 4] == e[1:5]) * (f[0] == e[0]) * (f[5] == e[5]):
                            return "No"

        return "Yes"

    print((f()))


problem_p02386()
