def problem_p00914():
    import itertools

    while True:

        N, K, S = list(map(int, input().split()))

        if N == K == S == 0:

            break

        ans = 0

        for l in itertools.combinations(list(range(1, N + 1)), K):

            if sum(l) == S:

                ans += 1

        print(ans)


problem_p00914()
