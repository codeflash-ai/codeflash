def problem_p02698():
    from bisect import bisect_left

    import sys

    sys.setrecursionlimit(10**7)

    input = sys.stdin.readline

    # スペース区切りの入力を読み込んで数値リストにして返します。

    def get_nums_l():

        return [int(s) for s in input().split(" ")]

    def log(*args):

        print("DEBUG:", *args, file=sys.stderr)

    INF = 999999999999999999999999

    n = int(input())

    A = [-INF] + get_nums_l()

    log(n)

    log(A)

    edges = [[] for _ in range(n + 1)]

    for _ in range(n - 1):

        # for line in map(lambda s: s.strip(), sys.stdin.readlines()):

        line = input().strip()

        log(line)

        u, v = map(int, line.split())

        edges[u].append(v)

        edges[v].append(u)

    def dfs(dp, u, p=None):

        # log(u)

        # log(dp)

        # dpの中でA[u]以上の値が入っている最小のindex

        i = bisect_left(dp, A[u])

        old = dp[i]

        dp[i] = A[u]

        # 有効値(INF以外)が入っている最大のindex

        ans[u] = bisect_left(dp, INF) - 1

        for v in edges[u]:

            if v == p:

                continue

            dfs(dp, v, u)

        # 巻き戻し

        dp[i] = old

    # ans[i] = ノード1からノードiへの経路中の最長増加部分列長

    ans = [0] * (n + 1)

    # dp[i] = 最長増加部分列長がiの場合の末尾ノード値の最小

    # dp[0]は使用しないので負の無限大を入れておく

    dp = [INF] * (n + 1)

    dp[0] = -INF

    dfs(dp, 1, None)

    for i in range(1, n + 1):

        print(ans[i])


problem_p02698()
