def problem_p02984():
    N = int(eval(input()))

    A = list(map(int, input().split()))

    x = [0 for _ in range(N)]

    for i in range(N):

        if i % 2 == 0:

            x[0] += A[i]

        else:

            x[0] -= A[i]

    x[0] //= 2

    for i in range(1, N):

        x[i] = A[i - 1] - x[i - 1]

    ans = " ".join(str(2 * X) for X in x)

    print(ans)


problem_p02984()
