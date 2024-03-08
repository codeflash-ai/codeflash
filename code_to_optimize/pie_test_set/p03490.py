def problem_p03490():
    def f(m, s, g):

        v, t = {s}, set()

        for i in m:

            for j in v:

                t.update((j + i, j - i))

            v, t = t, set()

        return g in v

    s = eval(input())

    x, y = list(map(int, input().split()))

    m = list(map(len, s.split("T")))

    if f(m[::2][1::], m[0], x) and f(m[1::2], 0, y):

        print("Yes")

    else:

        print("No")


problem_p03490()
