def problem_p00117():
    def f(x1, x2):

        W = [[i, 1001] for i in range(n)]

        W[x1] = [x1, 0]

        W[x2] = [x2, 2000]

        i = 0

        while 1:

            A = sorted(W, key=lambda x: x[1])

            p1, w1 = A[i]

            if p1 == x2:
                break

            for p2, w2 in A[i + 1 :]:

                way = (p1, p2)

                if way in cost:

                    tmp = w1 + cost[way]

                    W[p2] = [p2, min([w2, tmp])]

            i += 1

        return W[x2][1]

    n = int(input())

    m = int(input())

    cost = {}

    for _ in [0] * m:

        a, b, c, d = list(map(int, input().split(",")))

        cost[(a - 1, b - 1)] = c

        cost[(b - 1, a - 1)] = d

    x1, x2, y1, y2 = list(map(int, input().split(",")))

    x1 -= 1

    x2 -= 1

    print(y1 - y2 - f(x1, x2) - f(x2, x1))


problem_p00117()
