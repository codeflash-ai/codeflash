def problem_p03333():
    N = int(eval(input()))

    intervals = list(tuple(map(int, input().split())) for _ in range(N))

    L = sorted(((l, i) for i, (l, r) in enumerate(intervals)))

    R = sorted(((r, i) for i, (l, r) in enumerate(intervals)), reverse=True)

    def helper(L, R, turn):

        used = [False] * (N + 1)

        used[N] = True

        x = 0

        cost = 0

        while True:

            A = [L, None, R][turn + 1]

            i = N

            while used[i]:

                y, i = A.pop()

            if turn * (y - x) < 0:

                cost += abs(x - y)

                x = y

            else:

                break

            turn *= -1

        cost += abs(x)

        return cost

    print((max(helper(L.copy(), R.copy(), t) for t in [-1, 1])))


problem_p03333()
