def problem_p00966():
    from bisect import bisect

    n, a, b, q = map(int, input().split())

    W = [input().split() for i in range(a)]

    X = [int(x) for x, c in W]

    C = [c for x, c in W]

    P = [list(map(int, input().split())) for i in range(b)]

    Y = [y for y, h in P] + [n + 1]

    D = [0] * b

    for i in range(b):

        y0, h = P[i]
        y1 = Y[i + 1]

        l = y1 - y0

        D[i] = min(y0 - h, l)

    idx = 0

    S = {}

    for i in range(a):

        x = X[i]
        c = C[i]

        S[x] = c

        if x < Y[0]:

            continue

        while Y[idx + 1] <= x:
            idx += 1

        i = idx
        j = i

        while Y[0] <= x:

            while x < Y[i]:
                i -= 1

            y0, h = P[i]
            y1 = Y[i + 1]

            if h == 0:
                break

            x = h + ((x - y0) % D[i])

            assert x < y0

            S[x] = c

    def check(z):

        if z in S:

            return S[z]

        i = bisect(Y, z) - 1

        while Y[0] <= z:

            while z < Y[i]:
                i -= 1

            y0, h = P[i]
            y1 = Y[i + 1]

            if h == 0:
                break

            z = h + ((z - y0) % D[i])

            assert z < y0

            if z in S:

                return S[z]

        return "?"

    Z = [int(input()) for i in range(q)]

    print(*map(check, Z), sep="")


problem_p00966()
