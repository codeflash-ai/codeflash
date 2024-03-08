def problem_p02851():
    from collections import defaultdict

    N, K, *A = list(map(int, open(0).read().split()))

    S = [0] * (N + 1)

    for i in range(N):

        S[i + 1] = S[i] + A[i]

    d = defaultdict(int)

    ans = 0

    for j in range(N + 1):

        v = (S[j] - j) % K

        ans += d[v]

        d[v] += 1

        if j >= K - 1:

            d[(S[j - K + 1] - (j - K + 1)) % K] -= 1

    print(ans)


problem_p02851()
