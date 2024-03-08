def problem_p03685():
    import sys

    input = lambda: sys.stdin.readline().rstrip()

    R, C, N = list(map(int, input().split()))

    X = {0, R}

    Y = {0, C}

    Z = []

    for i in range(N):

        x1, y1, x2, y2 = list(map(int, input().split()))

        if (x1 == 0 or x1 == R or y1 == 0 or y1 == C) and (
            x2 == 0 or x2 == R or y2 == 0 or y2 == C
        ):

            Z.append((x1, y1, x2, y2))

            X.add(x1)

            X.add(x2)

            Y.add(y1)

            Y.add(y2)

    DX = {a: i for i, a in enumerate(sorted(list(X)))}

    DY = {a: i for i, a in enumerate(sorted(list(Y)))}

    R, C = DX[R], DY[C]

    def calc(a, b):

        if b == 0:

            return a

        if a == R:

            return b + R

        if b == C:

            return R + C + (R - a)

        if a == 0:

            return R + C + R + (C - b)

    A = []

    for i, (x1, y1, x2, y2) in enumerate(Z):

        x3, y3, x4, y4 = DX[x1], DY[y1], DX[x2], DY[y2]

        A.append((calc(x3, y3), i))

        A.append((calc(x4, y4), i))

    A = [l[1] for l in sorted(A, key=lambda x: x[0])]

    B = []

    while A:

        while len(B) and A[-1] == B[-1]:

            A.pop()

            B.pop()

        if A:

            B.append(A.pop())

    print(("NO" if len(B) else "YES"))


problem_p03685()
