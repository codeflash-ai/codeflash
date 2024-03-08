def problem_p03747():
    import sys

    input = sys.stdin.readline

    N, L, T = list(map(int, input().split()))

    XW = [[int(x) for x in input().split()] for _ in range(N)]

    X, W = list(zip(*XW))

    DX = [1 if w == 1 else -1 for w in W]

    Y = [(x + dx * T) % L for x, dx in zip(X, DX)]

    y0 = Y[0]

    Y.sort()

    # 番号1がすれ違わず進んだ場所に相当するアリの番号

    # すれ違うたびに1増える（W1=1） or 1減る（W1=-1）

    # T秒ちょうどもすれ違い終わったと考える

    x = 0

    x0, dx0 = X[0], DX[0]

    for y, dy in zip(X[1:], DX[1:]):

        if dx0 == dy:

            continue

        if dx0 == 1 and dy == -1:

            # 正の向きに追い越すので番号がひとつ増える

            x += (2 * T - (y - x0) - 1) // L + 1

        if dx0 == -1 and dy == 1:

            x -= (2 * T - (L + x0 - y)) // L + 1

    x %= N

    i = Y.index(y0)

    Y += Y

    answer = [None] * N

    answer[x:N] = Y[i : i + N - x]

    answer[0:x] = Y[i + N - x : i + N]

    print(("\n".join(map(str, answer))))


problem_p03747()
