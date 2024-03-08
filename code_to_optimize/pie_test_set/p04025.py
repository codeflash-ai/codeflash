def problem_p04025():
    # AtCoder Regular Contest 059

    # C - いっしょ

    # https://atcoder.jp/contests/arc059/tasks/arc059_a

    N = int(eval(input()))

    (*A,) = list(map(int, input().split()))

    mincost = 10**10

    for i in range(-100, 101):

        cost = 0

        for a in A:

            cost += (a - i) * (a - i)

        if cost < mincost:

            mincost = cost

    print(mincost)


problem_p04025()
