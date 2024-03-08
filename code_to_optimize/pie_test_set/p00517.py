def problem_p00517():
    W, H, N = list(map(int, input().split()))

    ans = 0

    X, Y = list(map(int, input().split()))

    for i in range(N - 1):

        X1, Y1 = list(map(int, input().split()))

        dX, dY = X1 - X, Y1 - Y

        if dX * dY > 0:

            ans += max(abs(dY), abs(dX))

        else:

            ans += abs(dY) + abs(dX)

        X, Y = X1, Y1

    print(ans)


problem_p00517()
