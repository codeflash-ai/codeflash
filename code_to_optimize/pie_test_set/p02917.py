def problem_p02917():
    N = int(eval(input()))

    B = list(map(int, input().split()))

    A = [0] * N

    A[0] = B[0]

    for i in range(1, N - 1):

        A[i] = min(B[i], B[i - 1])

    A[-1] = B[-1]

    print((sum(A)))


problem_p02917()
