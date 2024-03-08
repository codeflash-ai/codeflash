def problem_p03400():

    n = int(input())

    d, x = list(map(int, input().split()))

    ais = [int(input()) for _ in range(n)]

    def f(ai, d):

        return 1 + (d - 1) / (ai)

    print(x + sum([f(ai, d) for ai in ais] or [0]))


problem_p03400()
