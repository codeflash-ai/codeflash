def problem_p03357():
    from itertools import accumulate

    def solve(n, rev):

        def existence_right(rev_c):

            n2 = n * 2

            acc = [[0] * n2]

            row = [0] * n2

            for x in rev_c:

                row[n2 - x - 1] += 1

                acc.append(list(reversed(list(accumulate(row)))))

            return acc

        # How many white/black ball lower than 'k' righter than index x? (0<=x<=2N-1)

        # cost[color][k][x]

        cost = list(map(existence_right, rev))

        dp = [0] + list(accumulate(c[y] for y, c in zip(rev[1], cost[1])))

        for x, cw0, cw1 in zip(rev[0], cost[0], cost[0][1:]):

            ndp = [0] * (n + 1)

            cw0x = cw0[x]

            ndp[0] = prev = dp[0] + cw0x

            for b, (y, cb0, cb1) in enumerate(zip(rev[1], cost[1], cost[1][1:])):

                ndp[b + 1] = prev = min(dp[b + 1] + cw0x + cb1[x], prev + cw1[y] + cb0[y])

            dp = ndp

        return dp[n]

    n = int(eval(input()))

    # White/Black 'k' ball is what-th in whole row?

    rev = [[0] * n, [0] * n]

    for i in range(n * 2):

        c, a = input().split()

        a = int(a) - 1

        rev[int(c == "B")][a] = i

    print((solve(n, rev)))


problem_p03357()
