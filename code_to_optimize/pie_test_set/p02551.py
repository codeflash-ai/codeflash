def problem_p02551():
    N, Q = list(map(int, input().split()))

    N -= 2

    BIT = [[0 for i in range(N + 1)] for _ in range(2)]

    n = N

    def BIT_query(idx, BIT):

        res_sum = 0

        while idx > 0:

            res_sum += BIT[idx]

            idx -= idx & (-idx)

        return res_sum

    def BIT_update(idx, x, BIT):

        while idx <= n:

            BIT[idx] += x

            idx += idx & (-idx)

        return

    now = N * N

    max_x = [0, 0]

    for _ in range(Q):

        s, x = list(map(int, input().split()))

        s -= 1

        x -= 1

        x = N + 1 - x

        # print()

        depth = N + BIT_query(x, BIT[s])

        now -= depth

        # print(x, s, depth, now, BIT_query(x, BIT[s]), max_x[s])

        if x > max_x[s]:

            BIT_update(N - depth + 1, max_x[s] - x, BIT[1 - s])

            max_x[s] = x

    print(now)


problem_p02551()
