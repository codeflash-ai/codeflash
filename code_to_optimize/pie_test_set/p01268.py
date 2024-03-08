def problem_p01268():
    import itertools

    r = 110000

    s = int(r * 0.5)

    p = [1] * r

    p[0] = 0

    for i in range(1, s):

        if p[i]:

            p[2 * i + 1 :: i + 1] = [0 for x in range(2 * i + 1, r, i + 1)]

    while 1:

        N, P = list(map(int, input().split()))

        if N == -1:
            break

        i = N + 1

        prime = [0] * 23

        j = 0

        while j < 23:

            if p[i - 1] == 1:

                prime[j] = i

                j += 1

            i += 1

        wa = sorted(
            [sum(comb) for comb in itertools.combinations(prime, 2)] + [2 * a for a in prime]
        )

        print(wa[P - 1])


problem_p01268()
