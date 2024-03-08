def problem_p02695():
    import itertools

    N, M, Q = list(map(int, input().split()))

    buff = [list(map(int, input().split())) for _ in range(Q)]

    ans = 0

    for A in itertools.combinations_with_replacement(list(range(1, M + 1)), N):

        score = 0

        for a, b, c, d in buff:

            if A[b - 1] - A[a - 1] == c:

                score += d

        ans = max(ans, score)

    print(ans)


problem_p02695()
