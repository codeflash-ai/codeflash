def problem_p03599():
    A, B, C, D, E, F = list(map(int, input().split()))

    W = set()

    for a in range(0, F + 1, 100 * A):

        for b in range(0, F + 1, 100 * B):

            W.add(a + b)

    S = set()

    for c in range(0, F + 1, C):

        for d in range(0, F + 1, D):

            S.add(c + d)

    rate = -1

    for w in W:

        for s in S:

            if 0 < w + s <= F and s * 100 <= E * w:

                if s / (s + w) > rate:

                    rate = s / (s + w)

                    ans = w + s, s

    print((ans[0], ans[1]))


problem_p03599()
