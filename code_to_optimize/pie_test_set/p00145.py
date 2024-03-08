def problem_p00145():
    from functools import reduce

    def f(a, b):
        return [Card[a][0], Card[b][1]]

    n = int(input())

    Card = [[] for _ in [0] * n]

    Cost = {}

    for i in range(n):

        Card[i] = list(map(int, input().split()))

        Cost[(i, i)] = 0

    for i in range(1, n):

        for j in range(0, n - i):

            a = j + i

            Cost[(j, a)] = min(
                [
                    reduce(lambda a, b: a * b, f(j, k) + f(k + 1, a))
                    + Cost[(j, k)]
                    + Cost[(k + 1, a)]
                    for k in range(j, j + i)
                ]
            )

    print(Cost[0, n - 1])


problem_p00145()
