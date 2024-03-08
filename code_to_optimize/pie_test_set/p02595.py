def problem_p02595():
    from decimal import *

    getcontext().prec = 14

    N, D = list(map(int, input().split()))

    def distance(x1, x2, y1, y2):

        dx = x2 - x1

        dy = y2 - y1

        return (dx * dx + dy * dy).sqrt()

    X = [0] * N

    Y = [0] * N

    ans = 0

    for i in range(N):

        X[i], Y[i] = list(map(Decimal, input().split()))

        if distance(0, X[i], 0, Y[i]) <= D:

            ans += 1

    print(ans)


problem_p02595()
