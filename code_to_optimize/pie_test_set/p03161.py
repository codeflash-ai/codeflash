def problem_p03161():
    import sys

    INF = sys.maxsize

    N, K = (int(x) for x in input().split())

    heights = [int(x) for x in input().split()]

    memo = [INF] * N

    for i in range(N - 1, -1, -1):

        if i == N - 1:

            # ゴールからゴールの距離を0で初期化

            memo[i] = 0

            continue

        cost = INF

        for j in range(1, K + 1):

            # 範囲外

            if i + j >= N:

                continue

            cost = min(cost, memo[i + j] + abs(heights[i] - heights[i + j]))

        memo[i] = cost

    print((memo[0]))


problem_p03161()
