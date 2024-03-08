def problem_p03295():
    N, M = [int(t) for t in input().split()]

    R = []

    for i in range(M):

        a, b = [int(t) for t in input().split()]

        R.append((a, b))

    R.sort()

    inf = float("inf")

    Ra = [inf for a in range(N + 1)]

    for a, b in R:

        Ra[a] = min(Ra[a], b)

    R = [(a, Ra[a]) for a in range(1, N + 1) if Ra[a] != inf]

    R.sort(key=lambda r: r[1])

    bridge = [True] * (N)

    last = None

    for a, b in R:

        if last == None or last < a:

            bridge[b - 1] = False

            last = b - 1

    print((sum(1 for b in bridge if not b)))


problem_p03295()
