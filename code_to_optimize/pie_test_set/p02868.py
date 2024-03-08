def problem_p02868():
    import sys

    input = sys.stdin.readline

    sys.setrecursionlimit(10**7)

    N, M = list(map(int, input().split()))

    E = []

    for i in range(M):

        L, R, C = list(map(int, input().split()))

        E.append([L - 1, R - 1, C])

    E = sorted(E, key=lambda X: X[1])

    # N_はN以上の2べき数

    N_ = 2 ** (len(bin(N)) - 2)

    INF = 10**15  # 10**10

    seg = [INF] * (2 * N_ - 1)

    # k番目(0index)をaに更新

    def update(k, a):

        k += N_ - 1

        seg[k] = a

        while k > 0:

            k = (k - 1) // 2

            seg[k] = min(seg[k * 2 + 1], seg[k * 2 + 2])

    # 開区間！！！[a, b)の最小値を求める

    def query(a, b, k, l, r):

        if r <= a or b <= l:

            return INF

        if a <= l and r <= b:

            return seg[k]

        else:

            vl = query(a, b, k * 2 + 1, l, (l + r) // 2)

            vr = query(a, b, k * 2 + 2, (l + r) // 2, r)

            return min(vl, vr)

    # 初期化

    update(0, 0)

    for i in range(M):

        new = query(E[i][0], E[i][1], 0, 0, N_) + E[i][2]

        if new < seg[E[i][1] + N_ - 1]:

            update(E[i][1], new)

    if seg[N - 1 + N_ - 1] == INF:

        print((-1))

    else:

        print((seg[N - 1 + N_ - 1]))


problem_p02868()
