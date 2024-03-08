def problem_p03864():
    N, x = list(map(int, input().split()))

    A = list(map(int, input().split()))

    cnt = 0

    for i in range(1, N):

        diff = A[i] + A[i - 1] - x

        if diff > 0:

            cnt += diff

            A[i] -= min(A[i], diff)

    print(cnt)


problem_p03864()
