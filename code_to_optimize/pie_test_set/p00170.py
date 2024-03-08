def problem_p00170():
    def solve(placed, w1, w2):

        global weight, D

        n = len(N) - len(placed)

        x = list(set(N) - set(placed))

        if x == []:

            if weight > w1:

                D = placed

                weight = w1

            return

        for e in x:

            w = W[e]

            if w2 > S[e]:
                return

            a = w1 + w * n

            if a > weight:
                return

            b = w2 + w

            solve(placed + [e], a, b)

        return

    while 1:

        n = eval(input())

        if n == 0:
            break

        D = []

        weight = 1e9

        N = list(range(n))

        f = lambda x: [int(x[1]), int(x[2]), x[0]]

        x = [f(input().split()) for _ in [0] * n]

        W, S, Name = list(zip(*sorted(x)))

        solve([], 0, 0)

        for e in D[::-1]:

            print(Name[e])


problem_p00170()
