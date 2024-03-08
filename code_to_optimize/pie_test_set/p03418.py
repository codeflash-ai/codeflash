def problem_p03418():
    N, K = list(map(int, input().split()))

    if K == 0:

        print((N * N))

        exit()

    ans = 0

    for b in range(K + 1, N + 1):

        p = N // b

        ans += p * max(0, b - K) + max(0, N - p * b - K + 1)

    print(ans)


problem_p03418()
