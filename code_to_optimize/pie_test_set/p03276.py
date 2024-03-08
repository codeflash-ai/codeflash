def problem_p03276():
    import sys

    N, K = list(map(int, input().split()))

    X = [int(i) for i in input().split()]

    A = []

    B = []

    for x in X:

        if x < 0:

            A.append(x)

        else:

            B.append(x)

    ans = 10**18

    a = len(A)

    b = len(B)

    if a == 0:

        print((B[K - 1]))

        sys.exit()

    if b == 0:

        print((abs(A[-K])))

        sys.exit()

    for i in range(max(1, K - b), a + 1):

        if i >= K:

            ans = min(ans, abs(A[-i]))

            continue

        ans = min(ans, 2 * abs(A[-i]) + B[K - i - 1])

    for i in range(max(0, K - a - 1), b):

        if i >= K - 1:

            ans = min(ans, abs(B[i]))

            continue

        ans = min(ans, 2 * abs(B[i]) + abs(A[-K + i + 1]))

    print(ans)


problem_p03276()
