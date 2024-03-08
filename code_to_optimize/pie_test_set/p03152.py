def problem_p03152():
    P = 10**9 + 7

    N, M = list(map(int, input().split()))

    X = sorted([int(i) for i in input().split()])

    Y = sorted([int(i) for i in input().split()])

    A = [i for i in X]

    B = [i for i in Y]

    if len(set(A)) != N or len(set(B)) != M:

        print((0))

        exit()

    L = []

    x = 0

    y = 0

    D = N * M

    while len(A) > 0 and len(B) > 0:

        if max(A[-1], B[-1]) == D:

            if A[-1] > B[-1]:

                A.pop()

                L.append(y)

                x += 1

            elif A[-1] < B[-1]:

                B.pop()

                L.append(x)

                y += 1

            else:

                A.pop()

                B.pop()

                x += 1

                y += 1

            D -= 1

        else:

            C = max(A[-1], B[-1]) + 1

            tmp = x * y - (N * M - D)

            for i in range(C, D + 1)[::-1]:

                L.append(tmp)

                tmp -= 1

            D = max(A[-1], B[-1])

    if len(A) > 0:

        D = min(Y) - 1

        while len(A) > 0:

            C = A[-1] + 1

            tmp = x * y - (N * M - D)

            for i in range(C, D + 1)[::-1]:

                L.append(tmp)

                tmp -= 1

            D = A.pop() - 1

            x += 1

            L.append(M)

    if len(B) > 0:

        D = min(X) - 1

        while len(B) > 0:

            C = B[-1] + 1

            tmp = x * y - (N * M - D)

            for i in range(C, D + 1)[::-1]:

                L.append(tmp)

                tmp -= 1

            D = B.pop() - 1

            y += 1

            L.append(N)

    m = min(min(X), min(Y)) - 1

    for i in range(1, m + 1):

        L.append(i)

    ans = 1

    for i in L:

        ans *= i

        ans %= P

    print(ans)


problem_p03152()
