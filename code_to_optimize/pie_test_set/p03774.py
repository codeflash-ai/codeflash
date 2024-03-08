def problem_p03774():
    import numpy as np

    N, M = list(map(int, input().split()))

    human = []

    for i in range(N):

        a, b = list(map(int, input().split()))

        human.append((a, b))

    C = []

    for i in range(M):

        c, d = list(map(int, input().split()))

        C.append((c, d))

    for h in human:

        a, b = h

        now = 0

        mini = np.inf

        for i, ch in enumerate(C):

            c, d = ch

            L = abs(a - c) + abs(b - d)

            if mini > L:

                mini = L

                now = i + 1

        print(now)


problem_p03774()
