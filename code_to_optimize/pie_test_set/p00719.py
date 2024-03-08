def problem_p00719():
    while True:

        N, M, P, A, B = list(map(int, input().split()))

        if not (N | M | P | A | B):

            break

        A, B = A - 1, B - 1

        T = [int(x) for x in input().split()]

        dp = [[float("inf")] * M for _ in range(1 << N)]

        dp[0][A] = 0

        edges = []

        for _ in range(P):

            s, t, c = list(map(int, input().split()))

            s, t = s - 1, t - 1

            edges.append((s, t, c))

            edges.append((t, s, c))

        ans = float("inf")

        for state in range(1 << N):

            for s, t, c in edges:

                for k in range(N):

                    if state >> k & 1:

                        dp[state][t] = min(dp[state][t], dp[state & ~(1 << k)][s] + c / T[k])

        ans = min(dp[state][B] for state in range(1 << N))

        print(("Impossible" if ans == float("inf") else "{:.05f}".format(ans)))


problem_p00719()
