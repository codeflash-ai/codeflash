def problem_p02763():
    import sys

    from bisect import bisect_left, bisect_right, insort

    sr = lambda: sys.stdin.readline().rstrip()

    ir = lambda: int(sr())

    lr = lambda: list(map(int, sr().split()))

    N = ir()

    S = list("-" + sr())

    d = [[] for _ in range(26)]

    for i in range(1, N + 1):

        s = S[i]

        o = ord(s) - ord("a")

        d[o].append(i)

    Q = ir()

    for _ in range(Q):

        q, a, b = sr().split()

        if q == "1":

            a = int(a)

            if S[a] == b:

                continue

            prev = ord(S[a]) - ord("a")

            d[prev].remove(a)

            next = ord(b) - ord("a")

            insort(d[next], a)

            S[a] = b

        else:

            a = int(a)
            b = int(b)

            ans = 0

            for alpha in range(26):

                if bisect_right(d[alpha], b) - bisect_left(d[alpha], a) >= 1:

                    ans += 1

            print(ans)


problem_p02763()
