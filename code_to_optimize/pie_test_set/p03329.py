def problem_p03329(input_data):
    from bisect import bisect_right

    N = int(eval(input_data))

    unit = [1]

    for b in [6, 9]:

        u = b

        while u <= N:

            unit.append(u)

            u *= b

    unit.sort()

    Nu = len(unit)

    ans = N

    state_pool = [(0, 0, Nu - 1)]

    while state_pool:

        n, i, pk = state_pool.pop()

        if N - n >= (ans - i) * unit[pk]:

            continue

        sk = bisect_right(unit, N - n, 0, pk + 1)

        for k in range(sk):

            u = unit[k]

            c = n + u

            if c == N:

                if i + 1 < ans:

                    ans = i + 1

            else:

                state_pool.append((c, i + 1, k))

    return ans
