def problem_p03883():
    import sys

    readline = sys.stdin.readline

    readlines = sys.stdin.readlines

    import numpy as np

    N = int(readline())

    LR = [tuple(int(x) for x in line.split()) for line in readlines()]

    """
    
    ・奇数個のとき：中央にどれかを固定。
    
    ・偶数のとき：原点で左右に分けるとしてよい。
    
    ・基本的に長さで内側から昇順（奇数のときの中央は謎）。→ソートしてdpできる。
    
    ・左に置いた個数をもってdp更新。また、中央を使ったかも同時に持って。
    
    ・偶数のときは、中央に1つ置いた状態から始める。
    
    ・大きい区間から外側に入れていく。
    
    """

    LR.sort(key=lambda x: -x[1] - x[0])

    INF = 10**18

    dp_0 = np.full(N + 1, INF, np.int64)  # 左に置いた個数 → コスト。中央未確定。

    dp_1 = np.full(N + 1, INF, np.int64)  # 左に置いた個数 → コスト。中央確定。

    if N & 1:

        dp_0[0] = 0

    else:

        dp_1[0] = 0

    for n, (L, R) in enumerate(LR):

        length = R + L

        prev_0 = dp_0

        prev_1 = dp_1

        dp_0 = np.full(N + 1, INF, np.int64)

        dp_1 = np.full(N + 1, INF, np.int64)

        x = np.arange(n + 1, dtype=np.int64)  # もともと左側にあった個数

        # 左側に置く場合

        np.minimum(prev_0[: n + 1] + R + x * length, dp_0[1 : n + 2], out=dp_0[1 : n + 2])

        np.minimum(prev_1[: n + 1] + R + x * length, dp_1[1 : n + 2], out=dp_1[1 : n + 2])

        # 右側に置く場合

        np.minimum(prev_0[: n + 1] + L + (n - x) * length, dp_0[: n + 1], out=dp_0[: n + 1])

        np.minimum(prev_1[: n + 1] + L + (n - x) * length, dp_1[: n + 1], out=dp_1[: n + 1])

        # 中央に置く場合

        np.minimum(prev_0[: n + 1] + (N - 1) // 2 * length, dp_1[: n + 1], out=dp_1[: n + 1])

    answer = dp_1[N // 2]

    print(answer)


problem_p03883()
