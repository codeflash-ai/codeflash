def problem_p03743():
    import sys

    input = sys.stdin.readline

    """
    
    後ろから、到達不可能距離の集合を見ていきたい。
    
    集合を持つと厳しいが、最小値だけ持っておけばよい。
    
    """

    N, dist = list(map(int, input().split()))

    D = [int(x) for x in input().split()]

    # 各ターンの出発位置

    start_dist = [dist]

    for d in D:

        x = start_dist[-1]

        y = min(abs(x - d), x)

        start_dist.append(y)

    ng_dist = [None] * (N + 1)

    ng_dist[N] = 1

    for i, d in enumerate(D[::-1]):

        x = ng_dist[N - i]

        y = x if x <= d // 2 else x + d

        ng_dist[N - i - 1] = y

    eval(input())

    Q = [int(x) for x in input().split()]

    answer = ["YES" if ng_dist[d] <= start_dist[d - 1] else "NO" for d in Q]

    print(("\n".join(answer)))


problem_p03743()
