def problem_p03968():
    import sys

    input = sys.stdin.readline

    from collections import Counter, defaultdict

    import itertools

    from functools import lru_cache

    mask = (1 << 10) - 1

    symmetry = defaultdict(int)

    def encode(a, b, c, d):

        t = 1 << 40

        for x, y, z, w in [(a, b, c, d), (b, c, d, a), (c, d, a, b), (d, a, b, c)]:

            u = x + (y << 10) + (z << 20) + (w << 30)

            if t > u:

                t = u

        if a == b == c == d:

            symmetry[t] = 4

        elif a == c and b == d:

            symmetry[t] = 2

        else:

            symmetry[t] = 1

        return t

    @lru_cache(None)
    def decode(x):

        return [(x >> n) & mask for n in [0, 10, 20, 30]]

    N = int(eval(input()))

    tiles = []

    for _ in range(N):

        a, b, c, d = list(map(int, input().split()))

        tiles.append(encode(a, b, c, d))

    counter = Counter(tiles)

    P = [
        (1, n, n * (n - 1), n * (n - 1) * (n - 2), n * (n - 1) * (n - 2) * (n - 3))
        for n in range(N + 1)
    ]

    P += [(0, 0, 0, 0, 0)] * (N + 1)  # 負数代入対策

    power = [[n**e for e in range(5)] for n in range(5)]

    answer = 0

    for bottom, top in itertools.combinations(tiles, 2):

        counter[bottom] -= 1

        counter[top] -= 1

        a, b, c, d = decode(bottom)

        e, f, g, h = decode(top)

        for x, y, z, w in [(e, f, g, h), (f, g, h, e), (g, h, e, f), (h, e, f, g)]:

            # a,b,c,d

            # x,w,z,y

            tiles = [
                encode(p, q, r, s)
                for p, q, r, s in [(b, a, x, w), (c, b, w, z), (d, c, z, y), (a, d, y, x)]
            ]

            need = Counter(tiles)

            x = 1

            for tile, cnt in list(need.items()):

                x *= P[counter[tile]][cnt]  # 残っている個数、必要枚数

                x *= power[symmetry[tile]][cnt]

            answer += x

        counter[bottom] += 1

        counter[top] += 1

    # 上下を固定した分

    answer //= 3

    print(answer)


problem_p03968()
