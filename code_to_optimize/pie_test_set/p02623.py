def problem_p02623():
    N, M, K = list(map(int, input().split()))

    A = list(map(int, input().split()))

    B = list(map(int, input().split()))

    sumA = [0] * (N + 1)

    sumB = [0] * (M + 1)

    for i in range(N):

        sumA[i + 1] = sumA[i] + A[i]

    for i in range(M):

        sumB[i + 1] = sumB[i] + B[i]

    ans = 0

    import bisect

    for i in range(N + 1):

        if K - sumA[i] >= 0:

            idx = bisect.bisect_right(sumB, K - sumA[i])

            ans = max(ans, i + idx - 1)

    #         print(idx-1)

    print(ans)


problem_p02623()
