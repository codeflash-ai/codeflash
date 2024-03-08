def problem_p02804():
    import sys

    def input():
        return sys.stdin.readline().strip()

    def mapint():
        return list(map(int, input().split()))

    sys.setrecursionlimit(10**9)

    N, K = mapint()

    from collections import Counter

    As = list(mapint())

    count = Counter(As)

    mod = 10**9 + 7

    maxi = 0

    cum = 0

    pos = {}

    neg = {}

    pos[0] = 1

    neg[0] = 1

    As = sorted(list(set(As)))

    for i in range(1, 10**5 + 5):

        pos[i] = pos[i - 1] * i % mod

        neg[i] = pow(pos[i], mod - 2, mod)

    for a in As:

        cum += count[a]

        if cum < K:

            continue

        for i in range(1, min(K, count[a]) + 1):

            if cum - count[a] - K + i < 0:
                continue

            maxi += (
                a
                * pos[count[a]]
                * neg[i]
                * neg[count[a] - i]
                * pos[cum - count[a]]
                * neg[K - i]
                * neg[cum - count[a] - K + i]
            )

            maxi %= mod

    mini = 0

    cum = 0

    for a in As[::-1]:

        cum += count[a]

        if cum < K:

            continue

        for i in range(1, min(K, count[a]) + 1):

            if cum - count[a] - K + i < 0:
                continue

            mini += (
                a
                * pos[count[a]]
                * neg[i]
                * neg[count[a] - i]
                * pos[cum - count[a]]
                * neg[K - i]
                * neg[cum - count[a] - K + i]
            )

            mini %= mod

    print(((maxi - mini) % mod))


problem_p02804()
