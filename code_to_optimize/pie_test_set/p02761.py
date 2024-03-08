def problem_p02761():
    import sys

    import numpy as np

    N, M = [int(_) for _ in input().split()]

    if M == 0:

        if N == 1:

            print("0")

        else:

            ans = [0] * N

            ans[0] = 1

            print(("".join(map(str, ans))))

        sys.exit()

    S, C = np.array([[int(_) for _ in input().split()] for i in range(M)]).T

    ans = [-1] * N

    for i in range(M):

        j = int(S[i]) - 1

        if ans[j] == -1 or ans[j] == C[i]:

            ans[j] = C[i]

        else:

            print("-1")

            sys.exit()

    if N >= 2 and ans[0] == 0:

        print("-1")

        sys.exit()

    if N >= 2 and ans[0] == -1:

        ans[0] = 1

    for i in range(1, N):

        if ans[i] == -1:

            ans[i] = 0

    s = "".join(map(str, ans))

    print(s)


problem_p02761()
