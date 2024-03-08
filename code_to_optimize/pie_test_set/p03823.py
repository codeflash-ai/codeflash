def problem_p03823():
    import sys

    input = sys.stdin.readline

    from bisect import bisect_left, bisect_right

    INF = 10**18 + 100

    N, A, B = list(map(int, input().split()))

    S = [-INF] + [int(x) for x in sys.stdin.read().split()]

    MOD = 10**9 + 7

    dpX = [0] * (N + 1)  # 最後にYを選んだとして、直前に選んだXがどこにあるか

    dpY = [0] * (N + 1)  # 最後にXを選んだとして、直前に選んだYがどこにあるか

    dpX[0] = 1

    dpY[0] = 1

    dpX_cum = [1] * (N + 1) + [0]  # -1

    dpY_cum = [1] * (N + 1) + [0]  # -1

    dpX_left = 0

    dpY_left = 0

    for n, x in enumerate(S[2:], 2):

        iA = bisect_right(S, x - A)

        iB = bisect_right(S, x - B)

        # ....XY

        xy = dpY_cum[iB - 1] - dpY_cum[dpY_left - 1] if iB >= dpY_left else 0

        # ....YX

        yx = dpX_cum[iA - 1] - dpX_cum[dpX_left - 1] if iA >= dpX_left else 0

        # ....XX が不可能なら捨てる。明示的に捨てるのではなく、生きている番号だけ持つ

        if iA != n:

            dpY_left = n - 1

        if iB != n:

            dpX_left = n - 1

        dpX[n - 1] = xy

        dpX_cum[n - 1] = (dpX_cum[n - 2] + xy) % MOD

        dpX_cum[n] = dpX_cum[n - 1]

        dpY[n - 1] = yx

        dpY_cum[n - 1] = (dpY_cum[n - 2] + yx) % MOD

        dpY_cum[n] = dpY_cum[n - 1]

    answer = dpX_cum[N - 1] - dpX_cum[dpX_left - 1]

    answer += dpY_cum[N - 1] - dpY_cum[dpY_left - 1]

    answer %= MOD

    print(answer)


problem_p03823()
