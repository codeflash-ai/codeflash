def problem_p03479():
    X, Y = list(map(int, input().split()))

    t = X

    ans = 0

    while t <= Y:

        t *= 2

        ans += 1

    print(ans)


problem_p03479()
